import gzip
import json
import anny.anthropometry as a

p1 = r"D:\BarahaAvatar\.venv\Lib\site-packages\anny\data\mpfb2\mesh_metadata\basemesh_vertex_to_face_table.json.gz"
p2 = r"D:\BarahaAvatar\.venv\Lib\site-packages\anny\data\mpfb2\mesh_metadata\basemesh_face_to_vertex_table.json.gz"

vf = json.load(gzip.open(p1, "rt", encoding="utf-8"))
fv = json.load(gzip.open(p2, "rt", encoding="utf-8"))

waist = set(a.BASE_MESH_WAIST_VERTICES)

adj = {i: set() for i in range(len(vf))}

for face in fv:
    for x in face:
        for y in face:
            if x != y:
                adj[x].add(y)

dist = {i: 0 for i in waist}
queue = list(waist)
head = 0

while head < len(queue):
    u = queue[head]
    head += 1

    for n in adj[u]:
        if n not in dist:
            dist[n] = dist[u] + 1
            queue.append(n)

print("Waist:", len(waist))
print("Graph vertices reached:", len(dist))
print("Max distance:", max(dist.values()))
print(
    "Distances:",
    {d: sum(x == d for x in dist.values()) for d in range(0, 11)}
)
