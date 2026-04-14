import faiss
import numpy as np
import os
import pickle

class VectorDB:
    def __init__(self, dimension=384, index_path="data/faiss_index.bin"):
        """
        Dimension 384 corresponds to all-MiniLM-L6-v2.
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = index_path.replace(".bin", "_metadata.pkl")
        
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def add_documents(self, embeddings, texts):
        """
        embeddings: numpy array of vectors.
        texts: list of strings corresponding to the vectors.
        """
        if len(embeddings) == 0:
            return
            
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata.extend(texts)
        
        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def search(self, query_embedding, k=3):
        """
        Returns the top k most similar documents.
        """
        if self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.metadata):
                results.append(self.metadata[i])
        return results

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

if __name__ == "__main__":
    vdb = VectorDB()
    # Test would require embeddings, skipping for simple unit check
    print(f"Index total: {vdb.index.ntotal}")
