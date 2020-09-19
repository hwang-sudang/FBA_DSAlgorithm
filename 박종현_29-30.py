# 29. 1 그래프의 너비 우선 탐색
from collections import deque
from queue import Queue
import numpy as np


# 그래프의 인접리스트 표현
V = 10
adj = [[0] * V ] * V


# start에서 시작해 그래프를 너비 우선 탐색하고 각 정점의 방문 순서를 반환한다
def bfs(start):
    global adj
    # adj = np.array([[0] * V ] * V)   # 인접 리스트(행렬로 구현해 둠)
    discovered = [False] * len(adj)             # 각 정점의 방문 여부
    q = Queue()                                     # 방문할 정점 목록을 유지하는 큐
    order = []                                             # 정점의 방문 순서
    discovered[start] = True
    q.put(start)
    while not q.empty():
        here = q.get()
        order.append(here)
        for i in range(len(adj[here])):
            there = adj[here][i]
            if not discovered[there]:
                q.put(there)
                discovered[there] = True
    return order


# 29.2 최단 경로를 계산하는 너비 우선 탐색
# start에서 시작해 그래프를 너비 우선 탐색하고 시작점부터
# 각 정점까지의 최단거리와 너비 우선 탐색 스패닝 트리를 계산한다.
# distance[i] = start 부터 i까지의 최단 거리
# parent[i] = 너비 우선 탐색 스패닝 트리에서 i의 부모의 번호, 루트인 경우 자신의 번호.


def bfs2(start):
    global distance; distance = [-1] * len(adj)
    global parent; parent = [-1] * len(adj)
    # 방문할 정점 목록을 유지하는 큐
    q = deque()
    distance[start] = 0
    parent[start] = start
    q.appendleft(start)
    while len(q):
        here = q.popleft()
        # here의 모든 인접한 정점을 검사한다.
        for i in range(len(adj[here])):
            there = adj[here][i]
            # 처음 보는 정점이면 방문 목록에 집어넣는다.
            if distance[there] == -1:
                q.appendleft(there)
                distance[there] = distance[here] + 1
                parent[there] = here


def shortestPath(v):
    path = [v]
    while parent[v] != v:
        v = parent[v]
        path.append(v)
    path.reverse()
    return path


# 29.3 Sorting Game 문제를 해결하는 너비 우선 탐색 알고리즘
# perm을 정렬하기 위해 필요한 최소 뒤집기 연산의 수를 계산한다.
def bfs(perm):
    """
    perm: 순열
    """
    n = len(perm)
    # 목표 정점을 미리 계산한다.
    sorted_ = perm.copy()
    sorted_.sort()
    # 방문 목록(큐)과 시작점부터 각 정점까지의 거리
    q = deque()
    distance = dict()
    # 시작점을 큐에 넣는다
    distance = dict().fromkeys(perm, 0)
    q = deque(perm)
    while len(q):
        here = [q.popleft()]
        # 목표 정점을 발견했으면 곧장 종료한다.
        if here == sorted_:
            return distance[here]
        cost = distance[here]
        # 가능한 모든 부분 구간을 뒤집어 본다.
        for i in range(n):
            for j in range(i+2, n+1):
                here = here[ : i-1] + list(reversed(here[i-1: j])) + here[ j : ]
                if distance[here] == 0:
                    distance[here] = cost + 1
                    q.appendleft(here)
            here = here[ : i-1] + list(reversed(here[i-1: j])) + here[ j : ]
    return -1


# 29.4 Sorting Game을 더 빠르게 해결하는 너비 우선 탐색의 구현
toSort = {}     # key: vector, value = int
add = tuple.__add__


# [0, ..., n-1]의 모든 순열에 대해 toSort[]를 계산해 저장한다.
def precalc(n):
    perm = tuple(range(n))
    q = deque()
    q.append(perm)
    global toSort
    toSort = {perm: 0}
    while len(q):
        # queue에서 pop한 값을 here로 설정
        here = q.popleft()
        # toSort에서 key = here인 원소의 value를 cost로 설정
        cost = deque(v for k, v in toSort.items() if k == here)[0]
        for i in range(n):
            for j in range(i+1, n):
                here_r = add(add(here[:i], tuple(reversed(here[i: j+1]))), here[j+1:])
                if len([k for k, v in toSort.items() if k == here_r]) == 0:
                    toSort[here_r] = cost + 1
                    q.append(here_r)
                add(add(here[:i], tuple(reversed(here[i: j+1]))), here[j+1:])


def solve(perm):
    # perm을 [0, ..., n-1]의 순열로 변환한다.
    n = len(perm)
    precalc(n)
    fixed = [0] * n
    for i in range(n):
        smaller = 0
        for j in range(n):
            if perm[j] < perm[i]:
                smaller += 1
        fixed[i] = smaller
    return toSort[tuple(fixed)]


# 29.6 15-퍼즐을 해결하는 BFS
from queue import Queue


class State:
    # 인접한 상태들의 목록 반환
    def getAdjacent(self):
        return
    # map에 State를 넣기 위한 비교 연산자
    def lower(self, rhs):
        return
    # 종료 상태와 비교하기 위한 연산자
    def equal(self, rhs):
        return

# <C++> typedef map<State, int> StateMap
#  ; map<State, int>의 별칭으로서 StateMap
# start에서 finish까지 가는 최단 경로의 길이를 반환한다.
def bfs(start, finish):
    # 예외: start == finish
    if start == finish:
        return 0
    # 각 정점까지의 최단 경로의 길이를 저장
    c = {}
    # 앞으로 방문할 정점들을 저장한다.
    q = Queue()
    q.put(start)
    c[start] = 0
    # 너비 우선 탐색
    while not q.empty():
        here = q.get()
        cost = c[here]
        # 인접한 정점들의 번호를 얻어낸다
        adjacent = here.getAdjacent()
        for i in range(len(adjacent)):
            if c==1:
                pass





from queue import PriorityQueue

# 30.1 다익스트라의 최단 거리 알고리즘의 구현
# V: 정점의 개수
# 그래프의 인접리스트. (연결된 정점 번호, 간선 가중치) 쌍을 담는다.
# adj = [( , ), (, ) ...]

import sys
V = 8


def dijkstra(src):
    INF = sys.maxsize
    dist = [INF] * V
    dist[src] = 0
    pq = PriorityQueue()
    pq.put((0, src))
    while pq.qsize():
        pair = pq.get()
        cost, here = -pair[0], pair[1]
        # 만약 지금 꺼낸 것보다 더 짧은 경로를 알고 있다면 지금 꺼낸 것을 무시한다.
        if dist[here] < cost:
            pass
        for i in range(len(adj[here])):
            there = adj[here][i][0]
            nextDist = cost + adj[here][i][1]
            # 더 짧은 경로를 발견하면, dist를 갱신하고 우선 순위 큐에 넣는다
            if dist[there] > nextDist:
                dist[there] = nextDist
                pq.put((-nextDist, there))
    return dist