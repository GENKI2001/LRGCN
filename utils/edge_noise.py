from copy import deepcopy
import numpy as np
import torch


def add_edge_noise(
    data,
    noise_rate=0.0,
    noise_type="flip",
    random_seed=5,
    device="cpu",
):
    """
    エッジノイズを追加する関数
    元のdataオブジェクトは変更せず、コピーを返します。

    Parameters
    ----------
    data : torch_geometric.data.Data
        グラフデータ
    noise_rate : float
        ノイズ率 [0.0, 1.0]
        - flip: フリップするエッジの割合
        - delete: 削除するエッジの割合
        - add: 追加するエッジの割合（可能な非エッジのうち）
    noise_type : str
        ノイズタイプ: "flip" (削除+追加), "delete" (削除のみ), "add" (追加のみ)
    random_seed : int
        乱数シード
    device : str
        使用デバイス

    Returns
    -------
    data : torch_geometric.data.Data
        エッジノイズを追加したグラフデータ（コピー）
    stats : dict
        統計情報（削除されたエッジ、追加されたエッジなど）
    """
    assert 0.0 <= float(noise_rate) <= 1.0
    assert noise_type in ["flip", "delete", "add"], f"Unsupported noise_type: {noise_type}"
    
    # 常にコピーを作成してから変更
    data = deepcopy(data)
    
    if noise_rate == 0.0:
        stats = {
            "deleted_edges": np.array([], dtype=np.int64),
            "added_edges": np.array([], dtype=np.int64),
        }
        return data, stats

    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else edge_index.max().item() + 1
    num_edges = edge_index.size(1)
    
    rs = np.random.RandomState(int(random_seed))
    stats = {}

    if noise_type == "flip":
        # フリップ: 既存エッジの一部を削除し、同じ数の非エッジを追加
        k = int(np.floor(num_edges * float(noise_rate)))
        if k == 0:
            stats = {
                "deleted_edges": np.array([], dtype=np.int64),
                "added_edges": np.array([], dtype=np.int64),
            }
            return data, stats
        
        # 削除するエッジをランダムに選択
        edge_indices_to_remove = np.sort(rs.choice(num_edges, size=k, replace=False)).astype(np.int64)
        edges_to_remove = edge_index[:, edge_indices_to_remove]
        
        # エッジを削除
        mask = np.ones(num_edges, dtype=bool)
        mask[edge_indices_to_remove] = False
        remaining_edges = edge_index[:, torch.tensor(mask, device=device)]
        
        # 既存のエッジセットを作成（双方向）
        existing_edges_set = set()
        for i in range(remaining_edges.size(1)):
            src, dst = remaining_edges[0, i].item(), remaining_edges[1, i].item()
            existing_edges_set.add((src, dst))
            existing_edges_set.add((dst, src))  # 無向グラフを想定
        
        # 追加するエッジを生成（自己ループと既存エッジを除く）
        possible_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # 自己ループを除く
                if (i, j) not in existing_edges_set:
                    possible_edges.append((i, j))
        
        if len(possible_edges) < k:
            # 追加可能なエッジが少ない場合は、可能な分だけ追加
            k = len(possible_edges)
        
        if k > 0:
            added_edge_indices = rs.choice(len(possible_edges), size=k, replace=False)
            added_edges_list = [possible_edges[idx] for idx in added_edge_indices]
            
            # 双方向のエッジを追加
            added_edges_tensor = torch.zeros((2, k * 2), dtype=torch.long, device=device)
            for i, (src, dst) in enumerate(added_edges_list):
                added_edges_tensor[0, i * 2] = src
                added_edges_tensor[1, i * 2] = dst
                added_edges_tensor[0, i * 2 + 1] = dst
                added_edges_tensor[1, i * 2 + 1] = src
            
            # エッジを結合
            new_edge_index = torch.cat([remaining_edges, added_edges_tensor], dim=1)
        else:
            new_edge_index = remaining_edges
        
        stats["deleted_edges"] = edge_indices_to_remove
        stats["added_edges"] = np.array(added_edge_indices) if k > 0 else np.array([], dtype=np.int64)
        data.edge_index = new_edge_index
        
    elif noise_type == "delete":
        # 削除のみ: 既存エッジの一部を削除
        k = int(np.floor(num_edges * float(noise_rate)))
        if k == 0:
            stats = {
                "deleted_edges": np.array([], dtype=np.int64),
                "added_edges": np.array([], dtype=np.int64),
            }
            return data, stats
        
        edge_indices_to_remove = np.sort(rs.choice(num_edges, size=k, replace=False)).astype(np.int64)
        mask = np.ones(num_edges, dtype=bool)
        mask[edge_indices_to_remove] = False
        new_edge_index = edge_index[:, torch.tensor(mask, device=device)]
        
        stats["deleted_edges"] = edge_indices_to_remove
        stats["added_edges"] = np.array([], dtype=np.int64)
        data.edge_index = new_edge_index
        
    elif noise_type == "add":
        # 追加のみ: 非エッジをランダムに追加
        # 既存のエッジセットを作成（双方向）
        existing_edges_set = set()
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges_set.add((src, dst))
            existing_edges_set.add((dst, src))
        
        # 追加可能なエッジを生成（自己ループと既存エッジを除く）
        possible_edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # 自己ループを除く
                if (i, j) not in existing_edges_set:
                    possible_edges.append((i, j))
        
        if len(possible_edges) == 0:
            stats = {
                "deleted_edges": np.array([], dtype=np.int64),
                "added_edges": np.array([], dtype=np.int64),
            }
            return data, stats
        
        k = int(np.floor(len(possible_edges) * float(noise_rate)))
        if k == 0:
            stats = {
                "deleted_edges": np.array([], dtype=np.int64),
                "added_edges": np.array([], dtype=np.int64),
            }
            return data, stats
        
        added_edge_indices = rs.choice(len(possible_edges), size=k, replace=False)
        added_edges_list = [possible_edges[idx] for idx in added_edge_indices]
        
        # 双方向のエッジを追加
        added_edges_tensor = torch.zeros((2, k * 2), dtype=torch.long, device=device)
        for i, (src, dst) in enumerate(added_edges_list):
            added_edges_tensor[0, i * 2] = src
            added_edges_tensor[1, i * 2] = dst
            added_edges_tensor[0, i * 2 + 1] = dst
            added_edges_tensor[1, i * 2 + 1] = src
        
        # エッジを結合
        new_edge_index = torch.cat([edge_index, added_edges_tensor], dim=1)
        
        stats["deleted_edges"] = np.array([], dtype=np.int64)
        stats["added_edges"] = np.array(added_edge_indices)
        data.edge_index = new_edge_index

    return data, stats

