import os
from typing import Dict, List, Tuple, Any, Union
import cv2
import networkx as nx
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cdist, cosine
from tqdm import tqdm


class FaceEmbedder:
    def __init__(self, model=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model or InceptionResnetV1(pretrained="vggface2").eval().to(
            self.device
        )

    def load_image(self, image_path):
        """Load an image from disk and convert it to a tensor."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        # Optionally preprocess image if needed (resize, etc.)
        face_tensor = self.preprocess_face(image)
        return face_tensor

    def preprocess_face(self, face_image):
        """Preprocess the face image for embedding extraction (resize, normalize, etc.)."""
        face_image = cv2.resize(
            face_image, (160, 160)
        )  # Resize to 160x160 pixels if required
        face_tensor = (
            torch.tensor(face_image)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
        return face_tensor

    def get_face_embeddings(self, selected_frames, image_dir):
        """Get embeddings for each cropped face image."""
        face_embeddings = []

        # Initialize progress bar
        with tqdm(
            total=len(selected_frames), desc="Extracting Face Embeddings", unit="face"
        ) as pbar:
            for scene_id, faces in selected_frames.items():
                for face_data in faces:
                    embeddings = []
                    for frame_info in face_data["top_frames"]:
                        image_path = os.path.join(
                            image_dir, frame_info["image_path"]
                        )  # Construct full path to image
                        face_tensor = self.load_image(image_path)
                        with torch.no_grad():
                            embedding = self.model(face_tensor).cpu().numpy()
                        embeddings.append(
                            {
                                "frame_idx": frame_info["frame_idx"],
                                "embedding": embedding,
                                "image_path": frame_info["image_path"],
                            }
                        )

                    face_embeddings.append(
                        {
                            "scene_id": scene_id,
                            "unique_face_id": face_data["unique_face_id"],
                            "global_face_id": face_data["global_face_id"],
                            "embeddings": embeddings,  # Each embedding with its image path
                        }
                    )

                    # Update progress bar
                    pbar.update(1)

        return face_embeddings


class FaceClusterer:
    def __init__(self, similarity_threshold: float = 0.6, max_iterations: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations

    def build_graph(self, face_embeddings):
        """Builds a graph where nodes represent embeddings, and edges represent similarities."""
        G = nx.Graph()

        # Flatten embeddings with identifiers into node_data
        node_data = [
            (
                i,
                emb_info["embedding"],
                face_data,
                emb_info["frame_idx"],
                emb_info["image_path"],
            )
            for i, face_data in enumerate(face_embeddings)
            for emb_info in face_data["embeddings"]
        ]

        # Add nodes to the graph
        for i, (face_idx, embedding, face_data, frame_idx, image_path) in enumerate(
            node_data
        ):
            G.add_node(
                i,
                face_idx=face_idx,
                embedding=embedding,
                face_data=face_data,
                frame_idx=frame_idx,
                image_path=image_path,
            )

        # Add edges based on maximum similarity between embeddings
        with tqdm(
            total=len(node_data) * (len(node_data) - 1) // 2,
            desc="Building Graph",
            unit="edge",
        ) as pbar:
            for i in range(len(node_data)):
                for j in range(i + 1, len(node_data)):
                    similarity = 1 - cosine(
                        node_data[i][1].flatten(), node_data[j][1].flatten()
                    )  # Ensure embeddings are 1D
                    if similarity > self.similarity_threshold:
                        G.add_edge(i, j, weight=similarity)
                    pbar.update(1)

        return G, node_data

    def apply_chinese_whispers(self, G: nx.Graph) -> dict:
        """Runs the Chinese Whispers algorithm on the similarity graph with convergence check."""
        labels = {node: i for i, node in enumerate(G.nodes())}

        with tqdm(
            total=self.max_iterations, desc="Running Chinese Whispers", unit="iteration"
        ) as pbar:
            for iteration in range(self.max_iterations):
                nodes = list(G.nodes())
                np.random.shuffle(nodes)

                labels_changed = False
                for node in nodes:
                    neighbor_labels = [
                        labels[neighbor] for neighbor in G.neighbors(node)
                    ]
                    if neighbor_labels:
                        label_counts = np.bincount(neighbor_labels)
                        most_common_label = np.argmax(label_counts)
                        if labels[node] != most_common_label:
                            labels[node] = most_common_label
                            labels_changed = True

                # Check for convergence
                if not labels_changed:
                    print(f"Converged after {iteration + 1} iterations.")
                    break

                pbar.update(1)

        return labels

    def consolidate_clusters(self, initial_clusters: dict) -> dict:
        """Consolidate clusters by assigning each face to the best cluster among its initial assignments."""
        # Map to hold the best cluster assignment for each unique_face_id
        face_best_assignment: Dict[str, int] = {}
        face_embeddings: Dict[str, List[np.ndarray]] = {}
        face_data_map: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}

        # Step 1: Collect all clusters and embeddings for each unique_face_id
        for cluster_id, face_list in initial_clusters.items():
            for face_data in face_list:
                unique_face_id = face_data["unique_face_id"]
                embedding = face_data["embedding"]

                if unique_face_id not in face_embeddings:
                    face_embeddings[unique_face_id] = []
                    face_data_map[unique_face_id] = []
                face_embeddings[unique_face_id].append(embedding)
                face_data_map[unique_face_id].append((cluster_id, face_data))

        # Step 2: For each unique_face_id, consider only the clusters it was assigned to
        for unique_face_id, embeddings in face_embeddings.items():
            assigned_clusters = set(
                cluster_id for cluster_id, _ in face_data_map[unique_face_id]
            )
            print(
                f"Unique Face ID: {unique_face_id}, Assigned Clusters: {assigned_clusters}"
            )

            best_cluster_id: Union[int, None] = None
            max_avg_similarity = -1

            if len(assigned_clusters) == 1:
                best_cluster_id = next(iter(assigned_clusters))
            else:
                embeddings_array: np.ndarray = np.array(embeddings)
                if embeddings_array.ndim == 3 and embeddings_array.shape[1] == 1:
                    embeddings_array = embeddings_array.reshape(
                        embeddings_array.shape[0], embeddings_array.shape[2]
                    )

                if embeddings_array.ndim == 1:
                    embeddings_array = embeddings_array.reshape(1, -1)

                for cluster_id in assigned_clusters:
                    # Get embeddings of the cluster
                    cluster_embeddings = []
                    for face_data in initial_clusters[cluster_id]:
                        cluster_embeddings.append(face_data["embedding"])
                    cluster_embeddings_array: np.ndarray = np.array(cluster_embeddings)

                    if (
                        cluster_embeddings_array.ndim == 3
                        and cluster_embeddings_array.shape[1] == 1
                    ):
                        cluster_embeddings_array = cluster_embeddings_array.reshape(
                            cluster_embeddings_array.shape[0],
                            cluster_embeddings_array.shape[2],
                        )

                    if cluster_embeddings_array.ndim == 1:
                        cluster_embeddings_array = cluster_embeddings_array.reshape(
                            1, -1
                        )

                    # Step 3: Compute average similarity between face embeddings and cluster embeddings
                    similarities = 1 - cdist(
                        embeddings_array, cluster_embeddings_array, "cosine"
                    )
                    avg_similarity = np.mean(similarities)

                    if avg_similarity > max_avg_similarity:
                        max_avg_similarity = avg_similarity
                        best_cluster_id = cluster_id

            # Assign the face to the best cluster
            assert best_cluster_id is not None, "best_cluster_id should not be None"
            face_best_assignment[unique_face_id] = best_cluster_id

        # Step 4: Build consolidated clusters based on best assignments
        consolidated_clusters: Dict[int, List[Dict[str, Any]]] = {}
        for unique_face_id, best_cluster_id in face_best_assignment.items():
            if best_cluster_id not in consolidated_clusters:
                consolidated_clusters[best_cluster_id] = []

            # Add all face_data instances for this unique_face_id to the best cluster
            for cluster_id, face_data in face_data_map[unique_face_id]:
                # Avoid duplicates
                if face_data not in consolidated_clusters[best_cluster_id]:
                    consolidated_clusters[best_cluster_id].append(face_data)

        return consolidated_clusters

    def _max_similarity(self, face_list: list, embedding: np.ndarray) -> float:
        """Helper function to calculate the maximum similarity of an embedding with a list of faces."""
        similarities = similarities = [
            1 - cosine(face["embedding"].flatten(), embedding.flatten())
            for face in face_list
        ]
        return max(similarities)

    def cluster_faces(self, face_embeddings: list) -> dict:
        """Cluster faces based on their embeddings using Chinese Whispers."""
        G, node_data = self.build_graph(face_embeddings)
        labels = self.apply_chinese_whispers(G)

        initial_clusters: Dict[int, List[Dict[str, Any]]] = {}
        for node_idx, label in labels.items():
            face_data = node_data[node_idx][2]
            frame_idx = node_data[node_idx][3]
            image_path = node_data[node_idx][4]

            if label not in initial_clusters:
                initial_clusters[label] = []

            initial_clusters[label].append(
                {
                    "scene_id": face_data["scene_id"],
                    "unique_face_id": face_data["unique_face_id"],
                    "global_face_id": face_data["global_face_id"],
                    "frame_idx": frame_idx,
                    "image_path": image_path,  # Include image path in the cluster data
                    "embedding": node_data[node_idx][1],
                }
            )

        return self.consolidate_clusters(initial_clusters)
