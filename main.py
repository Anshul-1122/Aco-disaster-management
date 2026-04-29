import asyncio
import io
import json
import os
import zipfile
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="ACO Emergency Routing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request/Response models ----------

class NodeModel(BaseModel):
    id: str
    label: str
    type: str
    x: float
    y: float
    capacity: Optional[int] = None
    demand: Optional[int] = None
    priority: Optional[int] = None


class EdgeModel(BaseModel):
    id: str
    from_: str
    to: str
    distance: float
    travelTime: float
    roadCondition: str

    class Config:
        populate_by_name = True
        fields = {"from_": "from"}


class ACOParamsModel(BaseModel):
    numAnts: int = 20
    maxIterations: int = 100
    alpha: float = 1.0
    beta: float = 2.5
    evaporationRate: float = 0.3
    Q: float = 100.0
    initialPheromone: float = 1.0


class RunRequest(BaseModel):
    nodes: list[NodeModel]
    edges: list[EdgeModel]
    depotId: str
    targetNodes: list[str]
    params: ACOParamsModel


# ---------- ACO Core (NumPy) ----------

def build_adjacency(nodes: list[NodeModel], edges: list[EdgeModel]):
    node_ids = [n.id for n in nodes]
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    dist_matrix = np.full((n, n), np.inf)
    time_matrix = np.full((n, n), np.inf)
    blocked_matrix = np.ones((n, n), dtype=bool)

    node_priority = {}
    for node in nodes:
        node_priority[node.id] = node.priority or 1

    for edge in edges:
        fi = node_index.get(edge.from_)
        ti = node_index.get(edge.to)
        if fi is None or ti is None:
            continue
        if edge.roadCondition == "blocked":
            continue
        weight = 1.5 if edge.roadCondition == "damaged" else 1.0
        d = edge.distance * weight
        t = edge.travelTime * weight
        dist_matrix[fi][ti] = d
        dist_matrix[ti][fi] = d
        time_matrix[fi][ti] = t
        time_matrix[ti][fi] = t
        blocked_matrix[fi][ti] = False
        blocked_matrix[ti][fi] = False

    return dist_matrix, time_matrix, blocked_matrix, node_index, node_ids, node_priority


def compute_heuristic(time_matrix, node_priority, node_ids, n):
    eta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if not np.isinf(time_matrix[i][j]) and time_matrix[i][j] > 0:
                priority = node_priority.get(node_ids[j], 1)
                eta[i][j] = (1.0 / time_matrix[i][j]) * (1 + priority * 0.3)
    return eta


def select_next(current_idx, visited, pheromones, eta, blocked_matrix, n, alpha, beta):
    probs = np.zeros(n)
    for j in range(n):
        if j not in visited and not blocked_matrix[current_idx][j]:
            tau = max(pheromones[current_idx][j], 1e-10)
            probs[j] = (tau ** alpha) * (eta[current_idx][j] ** beta)

    total = probs.sum()
    if total <= 0:
        return None

    probs /= total
    return int(np.random.choice(n, p=probs))


def build_ant_route(depot_idx, target_indices, pheromones, eta, blocked_matrix, node_ids,
                    n, alpha, beta, dist_matrix, time_matrix, max_steps):
    path_indices = [depot_idx]
    visited = {depot_idx}
    visited_targets = set()
    total_dist = 0.0
    total_time = 0.0
    current = depot_idx
    steps = 0

    while len(visited_targets) < len(target_indices) and steps < max_steps:
        nxt = select_next(current, visited, pheromones, eta, blocked_matrix, n, alpha, beta)
        if nxt is None:
            break
        dist = dist_matrix[current][nxt]
        time = time_matrix[current][nxt]
        if np.isinf(dist):
            break
        total_dist += dist
        total_time += time
        path_indices.append(nxt)
        visited.add(nxt)
        current = nxt
        steps += 1
        if nxt in target_indices:
            visited_targets.add(nxt)

    ret_dist = dist_matrix[current][depot_idx]
    ret_time = time_matrix[current][depot_idx]
    if not np.isinf(ret_dist):
        total_dist += ret_dist
        total_time += ret_time
        path_indices.append(depot_idx)

    path_node_ids = [node_ids[i] for i in path_indices]
    visited_target_ids = [node_ids[i] for i in visited_targets]

    return {
        "path": path_node_ids,
        "totalDistance": round(float(total_dist), 2),
        "totalTime": round(float(total_time), 2),
        "nodesVisited": visited_target_ids,
        "feasible": len(visited_targets) == len(target_indices),
    }


def route_score(route):
    if not route["feasible"]:
        return float("inf")
    return route["totalDistance"] + route["totalTime"] * 2


def update_pheromones(pheromones, routes, node_index, evaporation_rate, Q, n):
    pheromones *= (1 - evaporation_rate)
    np.clip(pheromones, 1e-10, None, out=pheromones)
    for route in routes:
        if route["totalDistance"] <= 0:
            continue
        deposit = Q / route["totalDistance"]
        path = route["path"]
        for k in range(len(path) - 1):
            fi = node_index.get(path[k])
            ti = node_index.get(path[k + 1])
            if fi is not None and ti is not None:
                pheromones[fi][ti] += deposit
                pheromones[ti][fi] += deposit


def run_aco_generator(request: RunRequest):
    params = request.params
    dist_matrix, time_matrix, blocked_matrix, node_index, node_ids, node_priority = build_adjacency(
        request.nodes, request.edges
    )
    n = len(node_ids)
    depot_idx = node_index[request.depotId]
    target_indices = set(node_index[t] for t in request.targetNodes if t in node_index)

    eta = compute_heuristic(time_matrix, node_priority, node_ids, n)
    pheromones = np.full((n, n), params.initialPheromone)
    max_steps = n * 4

    global_best = None
    stagnation_count = 0
    last_best_score = float("inf")
    converged_at = None

    for iteration in range(params.maxIterations):
        routes = []
        for _ in range(params.numAnts):
            route = build_ant_route(
                depot_idx, target_indices, pheromones, eta,
                blocked_matrix, node_ids, n, params.alpha, params.beta,
                dist_matrix, time_matrix, max_steps
            )
            routes.append(route)

        routes.sort(key=route_score)
        elite_count = max(1, int(params.numAnts * 0.3))
        elite_routes = routes[:elite_count]

        update_pheromones(pheromones, elite_routes, node_index, params.evaporationRate, params.Q, n)

        iter_best = routes[0]
        if global_best is None or route_score(iter_best) < route_score(global_best):
            global_best = dict(iter_best)

        score = route_score(global_best)
        if abs(score - last_best_score) < 0.5:
            stagnation_count += 1
            if stagnation_count >= 15 and converged_at is None:
                converged_at = iteration
        else:
            stagnation_count = 0
        last_best_score = score

        pheromone_snapshot = {}
        for nid in node_ids:
            fi = node_index[nid]
            pheromone_snapshot[nid] = {
                other: round(float(pheromones[fi][node_index[other]]), 4)
                for other in node_ids
            }

        yield {
            "iteration": iteration,
            "bestRoute": global_best,
            "convergenceValue": round(score, 2) if score != float("inf") else None,
            "convergedAt": converged_at,
            "pheromoneMatrix": pheromone_snapshot,
            "done": False,
        }

    yield {
        "iteration": params.maxIterations - 1,
        "bestRoute": global_best,
        "convergenceValue": round(route_score(global_best), 2) if global_best else None,
        "convergedAt": converged_at,
        "pheromoneMatrix": {},
        "done": True,
    }


# ---------- Routes ----------

@app.get("/aco-api/health")
def health():
    return {"status": "ok", "engine": "python-numpy"}


@app.get("/aco-api/download")
def download_project():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        include_dirs = ["api", "src"]
        include_files = ["package.json", "vite.config.ts", "tsconfig.json", "index.html"]
        for d in include_dirs:
            full_dir = os.path.join(base, d)
            if os.path.exists(full_dir):
                for root, dirs, files in os.walk(full_dir):
                    for f in files:
                        abs_path = os.path.join(root, f)
                        arc_name = "aco-emergency-routing/" + os.path.relpath(abs_path, base)
                        zf.write(abs_path, arc_name)
        for f in include_files:
            abs_path = os.path.join(base, f)
            if os.path.exists(abs_path):
                zf.write(abs_path, "aco-emergency-routing/" + f)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=aco-emergency-routing.zip"},
    )


@app.post("/aco-api/run")
async def run_aco_stream(request: RunRequest):
    async def event_stream():
        loop = asyncio.get_event_loop()

        def generate_all():
            results = []
            for item in run_aco_generator(request):
                results.append(item)
            return results

        results = await loop.run_in_executor(None, generate_all)

        for item in results:
            yield f"data: {json.dumps(item)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
