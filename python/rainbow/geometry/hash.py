
import numpy as np

class HashEntry:
    def __init__(self, i, tri_idx, tet_idx, body_name, x, y, z):
        self.point_idx = i
        self.tri_idx = tri_idx
        self.tet_idx = tet_idx
        self.body = body_name
        self.verteies = (x, y, z)
    
    def __hash__(self) -> int:
        return hash((self.body, self.point_idx))

    def __eq__(self, o: object) -> bool:
        return (self.point_idx == o.point_idx and self.body == o.body)


class SpatialHash:
    def __init__(self, cell_size, num_tets=10000):
        self.cell_size = cell_size
        self.hash_table = {}
        self.table_size = num_tets + 99
        self.offset = 1000

    def compute_hash_value(self, x, y, z):
        p1, p2, p3 = 73856093, 19349663, 83492791
        hash_value = np.bitwise_xor(x * p1, np.bitwise_xor(y * p2, z * p3)) % self.table_size

        return hash_value
    
    def insert(self, point_idx, tri_idx, tet_idx, body_name, x, y ,z):
        i = int((x + self.offset) // self.cell_size)
        j = int((y + self.offset) // self.cell_size)
        k = int((z + self.offset) // self.cell_size)
        hash_value = self.compute_hash_value(i, j ,k)
        if hash_value not in self.hash_table:
            self.hash_table[hash_value] = set()
        self.hash_table[hash_value].add(HashEntry(point_idx, tri_idx, tet_idx, body_name, x, y, z))
    
    def query(self, x, y, z, body_name, tet_idx):
        hash_value = self.compute_hash_value(x, y, z)
        results = []
        if hash_value not in self.hash_table:
            return results
        for entry in self.hash_table[hash_value]:
            if entry.body == body_name:
                continue
            else:
                results.append(entry) # collision
        return results

    def clear(self):
        self.hash_table.clear()

    def set_hash_table_size(self, num_tets):
        self.table_size = num_tets + 99

    def set_cell_size(self, cell_size):
        self.cell_size = cell_size