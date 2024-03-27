#!python3
"""
get_depth.py

Gets the depth of a tree. The depth is the longest downward path.
"""

# A recursive function used by longestPath. See below
# link for details
# https:#www.geeksforgeeks.org/topological-sorting/
def topologicalSortUtil(v, Stack, visited, edges):
    visited[v] = True

    # Recur for all the vertices adjacent to this vertex
    # list<AdjListNode>::iterator i
    for i in edges[v]:
        if (not visited[i]):
            topologicalSortUtil(i, Stack, visited, edges)

    # Push current vertex to stack which stores topological
    # sort
    Stack.append(v)

# The function to find longest distances from a given vertex.
# It uses recursive topologicalSortUtil() to get topological
# sorting.
def paths(roots, edges, words):
    Stack = []
    visited = { word: False for word in words }
    dist = { word: -10**9 for word in words }

    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for word in words:
        if (visited[word] == False):
            topologicalSortUtil(word, Stack, visited, edges)
    # print(Stack)

    # Initialize distances to all vertices as infinite and
    # distance to source as 0
    for root in roots:
        dist[root] = 0
    # Stack.append(1)

    # Process vertices in topological order
    while (len(Stack) > 0):

        # Get the next vertex from topological order
        u = Stack[-1]
        Stack.pop()
        #print(u)

        # Update distances of all adjacent vertices
        # list<AdjListNode>::iterator i
        if (dist[u] != -10**9):
            for i in edges[u]:
                # print(u, i)
                if (dist[i] < dist[u] + 1):
                    dist[i] = dist[u] + 1

    return dist


def get_depth(pairs, print_near):
    edges = dict()
    words = set()
    for (A,B) in pairs:
        words.add(A)
        words.add(B)
        if A not in edges:
            edges[A] = []
        if B not in edges:
            edges[B] = []
        edges[B].append(A) # reverse of what is done elsewhere

    roots = set(words)
    for (A,B) in pairs:
        if A in roots and A != B:
            roots.remove(A)

    print(len(words), "words")
    print(len(roots), "roots")
    dist = paths(roots, edges, words)
    # Print calculated longest distances
    if print_near is not None:
        for w in words:
            if dist[w] <= print_near:
                print(w, dist[w])

    return max(dist[w] for w in words)

if __name__ == "__main__":
    import sys
    filename = sys.argv[1].strip()
    pairs = { tuple(line.strip().split(" ")) for line in open(filename, "r") if len(line) > 2 }
    print(len(pairs), "edges")
    print("Depth: ", get_depth(pairs, int(sys.argv[2]) if len(sys.argv) > 2 else None))


