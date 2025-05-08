"""Module for face embedding and clustering operations."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import networkx as nx
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cdist, cosine
from tqdm import tqdm

class FaceEmbedder:
    """Class for generating face embeddings from images."""

    def __init__(
        self,
        model: Optional[InceptionResnetV1] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the FaceEmbedder.

        Args:
            model: Pre-trained face recognition model.
            device: Computation device (CPU or GPU).
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model or InceptionResnetV1(pretrained="vggface2").eval().to(
            self.device
        )

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load an image from disk and convert it to a tensor."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")

        # Optionally preprocess image if needed (resize, etc.)
        face_tensor = self.preprocess_face(image)
        return face_tensor

    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess the face image for embedding extraction."""
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

    def get_face_embeddings(
        self, selected_frames_by_scene: Dict[str, List[Dict[str, Any]]], image_dir: str
    ) -> List[Dict[str, Any]]:
        """Get embeddings for each cropped face image, carrying along associated data.

        Args:
            selected_frames_by_scene: Output from FrameSelector, structured as 
                                     Dict[scene_id, List[unique_track_data]], where
                                     unique_track_data is Dict["unique_track_id", "top_frames"].
                                     Each item in "top_frames" has "image_path", 
                                     "face_mesh", "full_body_pose", etc.
            image_dir: Directory containing the cropped face images.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, one for each unique track.
                                  Each dict contains "unique_track_id", "scene_id", and 
                                  "frames_data" (a list of dicts with "frame_idx", 
                                  "embedding", "image_path", "face_mesh", "full_body_pose").
        """
        all_tracks_data_with_embeddings = []

        # Calculate total number of top_frames to process for tqdm
        total_top_frames = 0
        for scene_id, unique_tracks_in_scene in selected_frames_by_scene.items():
            for unique_track_data in unique_tracks_in_scene:
                total_top_frames += len(unique_track_data.get("top_frames", []))

        with tqdm(
            total=total_top_frames, desc="Extracting Face Embeddings", unit="frame"
        ) as pbar:
            for scene_id, unique_tracks_in_scene in selected_frames_by_scene.items():
                for unique_track_data in unique_tracks_in_scene:
                    track_unique_id = unique_track_data["unique_track_id"]
                    
                    current_track_frames_data = []
                    for frame_info in unique_track_data.get("top_frames", []):
                        image_path_full = os.path.join(
                            image_dir, frame_info["image_path"]
                        )
                        try:
                            face_tensor = self.load_image(image_path_full)
                            with torch.no_grad():
                                embedding = self.model(face_tensor).cpu().numpy()
                        except ValueError as e:
                            print(f"Skipping embedding for {image_path_full} due to load error: {e}")
                            pbar.update(1)
                            continue # Skip this frame if image can't be loaded
                            
                        current_track_frames_data.append(
                            {
                                "frame_idx": frame_info["frame_idx"],
                                "embedding": embedding,
                                "image_path": frame_info["image_path"],
                                "face_mesh": frame_info.get("face_mesh"), # Pass through
                                "full_body_pose": frame_info.get("full_body_pose"), # Pass through
                                "face_coord": frame_info.get("face_coord"), # Pass through, might be useful
                                "total_score": frame_info.get("total_score") # Pass through quality score
                            }
                        )
                        pbar.update(1)
                    
                    if current_track_frames_data: # Only add track if it has successfully processed frames
                        all_tracks_data_with_embeddings.append(
                            {
                                "unique_track_id": track_unique_id,
                                "scene_id": scene_id,
                                "frames_data": current_track_frames_data,
                            }
                        )
        return all_tracks_data_with_embeddings

class FaceClusterer:
    """Class for clustering face embeddings using Chinese Whispers algorithm."""

    def __init__(
        self, similarity_threshold: float = 0.6, max_iterations: int = 100
    ) -> None:
        """Initialize the FaceClusterer.

        Args:
            similarity_threshold: Threshold for considering two faces similar.
            max_iterations: Maximum number of iterations for Chinese Whispers.
        """
        self.similarity_threshold = similarity_threshold
        self.max_iterations = max_iterations

    def build_graph(
        self, all_tracks_data: List[Dict[str, Any]]
    ) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
        """Build a graph of embeddings and similarities.

        Nodes represent individual frame embeddings. Edges represent similarities.

        Args:
            all_tracks_data: List of track data from FaceEmbedder.
                             Each dict has "unique_track_id", "scene_id", and "frames_data".
                             "frames_data" is a list of dicts with "embedding", "frame_idx", 
                             "image_path", "face_mesh", "full_body_pose", etc.

        Returns:
            The constructed graph and a flat list of node_attributes (one per embedding/frame).
        """
        G = nx.Graph()
        node_attributes_list = [] # Flat list, each item corresponds to a node (an embedding)
        node_idx_counter = 0

        for track_data in all_tracks_data:
            unique_track_id = track_data["unique_track_id"]
            scene_id = track_data["scene_id"]
            for frame_embedding_data in track_data["frames_data"]:
                node_attributes = {
                    "node_id": node_idx_counter,
                    "unique_track_id": unique_track_id,
                    "scene_id": scene_id,
                    **frame_embedding_data # Merges all keys from frame_embedding_data
                }
                node_attributes_list.append(node_attributes)
                
                G.add_node(
                    node_idx_counter,
                    attr_dict=node_attributes # Store all attributes directly on the node
                )
                node_idx_counter += 1
        
        # Add edges based on similarity between embeddings of different frames
        # Note: This compares every frame embedding with every other frame embedding.
        # For N total frame embeddings, this is O(N^2) comparisons.
        with tqdm(
            total=len(node_attributes_list) * (len(node_attributes_list) - 1) // 2,
            desc="Building Clustering Graph",
            unit="edge"
        ) as pbar:
            for i in range(len(node_attributes_list)):
                for j in range(i + 1, len(node_attributes_list)):
                    # embedding is expected to be a numpy array, ensure it's flattened if multi-dimensional.
                    embedding_i = node_attributes_list[i]["embedding"].flatten()
                    embedding_j = node_attributes_list[j]["embedding"].flatten()
                    
                    similarity = 1 - cosine(embedding_i, embedding_j)
                    if similarity > self.similarity_threshold:
                        G.add_edge(node_attributes_list[i]["node_id"], node_attributes_list[j]["node_id"], weight=similarity)
                    pbar.update(1)

        return G, node_attributes_list

    def apply_chinese_whispers(self, G: nx.Graph) -> Dict[Any, int]:
        """Apply the Chinese Whispers algorithm to cluster the graph.

        This method implements the Chinese Whispers algorithm, an approach to
        graph-based clustering. It iteratively updates node labels based on
        the most common label among their neighbors.

        Args:
            G (nx.Graph): The input graph to be clustered.

        Returns:
            Dict[Any, int]: A dictionary mapping each node to its cluster label.
        """
        labels: Dict[Any, int] = {node: i for i, node in enumerate(G.nodes())}

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
                        most_common_label = int(np.argmax(label_counts))
                        if labels[node] != most_common_label:
                            labels[node] = most_common_label
                            labels_changed = True

                if not labels_changed:
                    print(f"Converged after {iteration + 1} iterations.")
                    break

                pbar.update(1)

        return labels

    def consolidate_clusters(self, initial_clusters_by_node_id: Dict[int, List[int]], node_attributes_list: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Consolidates clusters by re-mapping node_ids to their full attributes.
        The core logic of `consolidate_clusters` regarding averaging similarities for 
        re-assignment based on unique_track_id is removed, as clustering is now done 
        directly on frame embeddings. Each node is an embedding.
        The main purpose now is to structure the output nicely.

        Args:
            initial_clusters_by_node_id: Dict mapping cluster_label to list of node_ids in that cluster.
            node_attributes_list: Flat list of attributes for each node (embedding).

        Returns:
            Dict[int, List[Dict[str, Any]]]: Consolidated clusters, where keys are cluster_labels
                                            and values are lists of full attribute dicts for each member.
        """
        final_clusters: Dict[int, List[Dict[str, Any]]] = {}
        for cluster_label, node_ids_in_cluster in initial_clusters_by_node_id.items():
            if cluster_label not in final_clusters:
                final_clusters[cluster_label] = []
            for node_id in node_ids_in_cluster:
                # Find the node_attributes by node_id
                # This assumes node_attributes_list is indexed by node_id if node_id starts from 0
                # Or, create a map if node_ids are not strictly sequential from 0
                node_attr = next((attr for attr in node_attributes_list if attr["node_id"] == node_id), None)
                if node_attr:
                    final_clusters[cluster_label].append(node_attr)
        return final_clusters

    def _max_similarity(
        self, face_list: List[Dict[str, Any]], embedding: np.ndarray
    ) -> float:
        """Calculate max similarity of an embedding with a list of faces.

        Args:
            face_list: List of face data.
            embedding: The embedding to compare.

        Returns:
            The maximum similarity.
        """
        similarities = [
            1 - cosine(face["embedding"].flatten(), embedding.flatten())
            for face in face_list
        ]
        return max(similarities)

    def cluster_faces(self, all_tracks_data: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        """Cluster faces based on their embeddings using Chinese Whispers.
        Each node in the graph is now a single frame embedding.
        """
        G, node_attributes_list = self.build_graph(all_tracks_data)
        if not G.nodes():
            print("No nodes in graph to cluster. Returning empty clusters.")
            return {}
            
        node_id_to_cluster_label = self.apply_chinese_whispers(G) # Returns {node_id: cluster_label}

        # Group node_ids by their cluster_label
        clusters_by_label: Dict[int, List[int]] = {}
        for node_id, label in node_id_to_cluster_label.items():
            if label not in clusters_by_label:
                clusters_by_label[label] = []
            clusters_by_label[label].append(node_id)

        # The old `consolidate_clusters` was more complex due to a different input structure.
        # Now, it primarily re-maps node_ids back to their full data.
        # The name "consolidate_clusters" might be a bit strong for its new role, 
        # but we keep it for consistency unless a major rewrite of that part is done.
        return self.consolidate_clusters(clusters_by_label, node_attributes_list)
