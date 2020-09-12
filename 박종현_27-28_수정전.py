# 그래프의 인접 리스트
# 각 정점을 방문했는지 여부를 나타내는 리스트


def dfs(here):
    """
    깊이 우선 탐색을 구현한다.
    here: int
    """
    print("DFS Visits", here)
    visited[here] = True
    # 모든 인접 정점을 순회하면서
    for i in range(len(adj[here])):
        there = adj[here][i]
    # 아직 방문한 적이 없다면 방문한다.
        if not visited[there]:
            dfs(there)
    return


def dfsAll():
    """
    모든 정점을 방문한다.
    """
    visited = [False] * len(adj)  # visited를 모두 false로 초기화한다.
    for i in range(len(adj)):  # 모든 정점을 순회하면서, 아직 방문한 적 없으면 방문한다.
        if not visited[i]:
            dfs(i)
    return


def substract(a, b):
    return "".join(a.rsplit(b))


# 그래프(인접 행렬) 생성
# 알파벳의 각 글자에 대한 인접행렬 표현
# 간선 (i, j)는 i가 j보다 앞에 와야 함을 의미한다.


def makeGraph(words):
    """
    words: vector
    """
    adj = [[0] * 26] * 26
    for j in range(1, len(words)):
        i = j - 1
        len = min(len(words[i]), len(words[j]))
        for k in range(len):
            if words[i][k] != words[j][k]:
                a = words[i][k] - 'a'
                b = words[j][k] - 'a'
                adj[a][b] = 1
                break
    return


# 깊이 우선 탐색을 이용한 위상 정렬
seen = []
order = []


def dfs(here):
    seen[here] = 1
    for there in range(len(adj)):
        if adj[here][there] and not seen[there]:
            dfs(there)
    order.append(here)


# adj에 주어진 그래프를 위상정렬한 결과를 반환한다.
# 그래프에 DAG가 아니라면 빈 벡터를 반환한다.
def topologicalSort():
    """
    return : vector
    """
    m = len(adj)
    seen = [0] * m
    order.clear()
    for i in range(m):
        if not seen[i]:
            dfs(i)
            order.reverse()
    # 만약 그래프가 DAG가 아니라면 정렬 결과에 역방향 간선이 있다.
    for i in range(m):
        for j in range(i + 1, m):
            if adj[order[j]][order[i]]:
                return []
    # 없는 경우라면 DFS에서 얻은 순서를 반환한다.
    return order


# 고대어 사전
C = int(input())
for _ in range(C):
    words = []
    n = int(input())
    for _ in range(n):
        words.append(input())

    makeGraph(words)
    result = topologicalSort()

    if len(result) == 0:
        print("INVALID HYPOTHESIS")
    else:
        for i in range(len(result)):
            print(str(result[i] + 'a'))

# 단어 제한 끝말잇기
import sys

input = sys.stdin.readline
C = int(input())
for _ in range(C):
    n = int(input())
    words = []
    for _ in range(n):
        words.append(input())

#  28.4 깊이 우선 탐색을 이용한 오일러 서킷 찾기
adj = [[0] * 26] * 26



def getEulerCircuit(here, circuit):
    """
    here: int
    circuit: list of integers
    """
    for there in range(len(adj[here])):
        while adj[here][there] > 0:
            adj[here][there] -= 1
            adj[there][here] -= 1
            getEulerCircuit(there, circuit)
    circuit.append(here)
    return


# 28.5 끝말잇기 문제의 입력을 그래프로 만들기
adj = [[0] * 26] * 26  # adj[i][j] = i와 j 사이의 간선의 수
graph = [[''] * 26] * 26  # graph[i][j] = i로 시작해서 j로 끝나는 단어의 목록
indegree, outdegree = [0] * 26, [0] * 26


def makeGraph(words):
    """
    words:
    """
    # 전역 변수 초기화
    global adj, graph, indegree, outdegree
    for i in range(26):
        for j in range(26):
            graph[i][j] = ''
    adj = [[0] * 26] * 26
    graph = [[''] * 26] * 26
    indegree, outdegree = [0] * 26, [0] * 26
    # 각 단어를 그래프에 추가한다.
    for i in range(len(words)):
        a = ord(words[i][0]) - ord(a)
        b = ord(words[i][len(words[i]) - 1]) - ord(a)
        graph[a][b].append(words[i])
        adj[a][b] += 1
        outdegree[a] += 1
        indegree[b] += 1
    return


# 28.6 방향 그래프에서 오일러 서킷 혹은 트레일을 찾아내기
def getEulerCircuit(here, circuit):
    """
    유향 그래프의 인접행렬 adj가 주어질 때 오일러 서킷 혹은 트레일을 계산한다.
    here: int
    circuit: list of integers
    """
    for there in range(len(adj)):
        while adj[here][there] > 0:
            adj[here][there] -= 1
            getEulerCircuit(there, circuit)
    circuit.append(here)
    return


def getEulerTrailorCircuit():
    """
   현재 그래프의 오일러 트레일이나 서킷 반환한다.
    """
    global circuit; circuit = []
    # 우선 트레일을 찾아본다: 시작점이 존재하는 경우
    for i in range(26):
        if outdegree[i] == indegree[i] + 1:
            getEulerCircuit(i, circuit)
            return circuit
    # 아니면 서킷이니, 간선에 인접한 아무 정점에서나 시작한다.
    for i in range(26):
        if outdegree[i]:
            getEulerCircuit(i, circuit)
            return circuit
    # 모두 실패한 경우 빈 배열을 반환한다.
    return circuit


# 28.7 끝말잇기 문제를 오일러 트레일 문제로 바꾸어 해결하는 알고리즘
def checkEuler():
    """
    현재 그래프의
    """
    # 예비 시작점과 끝점의 수
    plus1, minus1 = 0, 0
    for i in range(26):
        delta = outdegree[i] - indegree[i]
        # 모든 정점의 차수는 -1, 0, 1 중 하나.
        if (delta < -1) or (1 < delta):
            return False
        if delta == 1:
            plus1 += 1
        if delta == -1:
            minus1 += 1
    # 시작점과 끝점은 각 하나씩 있거나 하나도 없어야 한다.
    return (plus1 == 1 & minus1 == 1) | (plus1 == 0 & minus1 == 0)


def solve(words):
    makeGraph(words)
    # 차수가 맞지 않으면 실패
    if not checkEuler():
        return "IMPOSSIBLE"
    # 오일러 서킷이나 경로를 찾아낸다
    circuit = getEulerTrailorCircuit()
    # 모든 간선을 방문하지 못햇으면 실패
    if len(circuit) != len(words) + 1:
        return "IMPOSSIBLE"
    # 아닌 경우 방문 순서를 뒤집은 뒤 간선들을 모아 문자열로 만들어 반환한다.
    circuit.reverse()
    ret = ""
    for i in range(len(circuit)):
        a = circuit[i-1]
        b = circuit[i]
        if len(ret):
            ret += " "
        ret += graph[a][b][-1]
        graph[a][b].pop()
    return ret