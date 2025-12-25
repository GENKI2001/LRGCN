
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree, add_self_loops, to_dense_adj


def make_undirected(data: Data, device: torch.device):
    """
    データオブジェクトのedge_indexを無向グラフに修正し、adj2を追加する関数
    
    Args:
        data: PyTorch Geometricのデータオブジェクト
        device: 計算デバイス
    
    Returns:
        Data: 無向グラフに修正され、adj2が追加されたデータオブジェクト
    """
    
    # 無向グラフに変換（(u,v)と(v,u)を同じエッジとして扱う）
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes).to(device)
    
    try:
        # 正規化された隣接行列を作成
        adjacency_matrices = create_normalized_adjacency_matrices(data, device, max_hops=2)
        
        # 各hopの隣接行列をdataに追加
        for hop in range(1, 3):
            adj_key = f'adj_{hop}hop'
            if adj_key in adjacency_matrices:
                # adj1とadj2はそのまま、adjはadj1のエイリアスとして設定
                setattr(data, f'adj{hop}', adjacency_matrices[adj_key])
                if hop == 1:
                    setattr(data, 'adj', adjacency_matrices[adj_key])
            else:
                print(f"警告: {adj_key}の生成に失敗しました。メモリ不足の可能性があります。")
            
    except Exception as e:
        print(f"警告: adj2の生成中にエラーが発生しました: {e}")
    
    return data



def create_normalized_adjacency_matrices(data, device, max_hops=2):
    """
    H2GCN スタイル: 各 k について「exactly k-hop」だけを含む隣接を作る。
    1-hop は自己ループあり・対称正規化、k>=2 も各 k-hop グラフ上で対称正規化。

    実装方針:
      1) 無向・自己ループありの A1（0/1, dense bool）を作る
      2) A_k_bool = (A_{k-1}_bool @ A1_bool) > 0 で到達可能性を拡張（dense bool）
      3) exact_k = A_k_bool から、すべての {1..k-1} hop を論理的に除去
      4) exact_k をスパースにして D^{-1/2} A D^{-1/2} で正規化
    """
    out = {}

    # --- 1-hop（無向・自己ループあり） ---
    edge_index_undirected = to_undirected(data.edge_index, num_nodes=data.num_nodes).to(device)
    # 自己ループを追加
    edge_index_with_loops, _ = add_self_loops(edge_index_undirected, num_nodes=data.num_nodes)
    edge_index_with_loops = edge_index_with_loops.to(device)
    row, col = edge_index_with_loops

    # 1-hop の対称正規化
    deg_1hop = degree(row, num_nodes=data.num_nodes, dtype=torch.float32).to(device)
    deg_inv_sqrt_1hop = torch.pow(deg_1hop, -0.5)
    deg_inv_sqrt_1hop[torch.isinf(deg_inv_sqrt_1hop)] = 0.0
    norm_1hop = deg_inv_sqrt_1hop[row.to(device)] * deg_inv_sqrt_1hop[col.to(device)]

    adj_1hop = torch.sparse_coo_tensor(
        indices=edge_index_with_loops.to(device),
        values=norm_1hop.to(device),
        size=(data.num_nodes, data.num_nodes),
        device=device,
        dtype=torch.float32,
    ).coalesce()
    out["adj_1hop"] = adj_1hop

    if max_hops == 1:
        return out

    # 0/1 の dense bool A1 を一度だけ作成（OOM の可能性があるため巨大グラフでは注意）
    A1_bool = torch.sparse_coo_tensor(
        indices=edge_index_with_loops.to(device),
        values=torch.ones(edge_index_with_loops.size(1), device=device),
        size=(data.num_nodes, data.num_nodes),
        device=device,
    ).to_dense().bool()

    # 既知 hop の集合（論理和）を持っておく
    reached_bool = A1_bool.clone()

    # A_k_bool を順に構築して exact_k を取り出す
    Ak_bool = A1_bool
    for k in range(2, max_hops + 1):
        # Ak = A_{k-1} * A1 （bool 行列積を「通常の matmul → >0」で近似）
        Ak_bool = (Ak_bool.float().mm(A1_bool.float()) > 0)

        # exactly k-hop = Ak_bool から 1..(k-1)hop を除去
        exact_k_bool = Ak_bool & (~reached_bool)

        # スパース化
        idx = exact_k_bool.nonzero(as_tuple=False).T.to(device)  # [2, nnz]
        if idx.numel() == 0:
            # 空でもキーを作っておく（好みでスキップ可）
            out[f"adj_{k}hop"] = torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long, device=device),
                values=torch.empty((0,), dtype=torch.float32, device=device),
                size=(data.num_nodes, data.num_nodes),
                device=device,
                dtype=torch.float32,
            )
        else:
            r, c = idx
            deg_k = degree(r, num_nodes=data.num_nodes, dtype=torch.float32).to(device)
            deg_inv_sqrt_k = torch.pow(deg_k, -0.5)
            deg_inv_sqrt_k[torch.isinf(deg_inv_sqrt_k)] = 0.0

            norm_k = deg_inv_sqrt_k[r] * deg_inv_sqrt_k[c]
            adj_k = torch.sparse_coo_tensor(
                indices=idx,
                values=norm_k,
                size=(data.num_nodes, data.num_nodes),
                device=device,
                dtype=torch.float32,
            ).coalesce()
            out[f"adj_{k}hop"] = adj_k

        # 既知集合を更新（次の k+1 で除去に使う）
        reached_bool |= exact_k_bool

    return out

def get_adjacency_matrix(adjacency_matrices: dict, hop: int):
    """
    指定されたhop数の隣接行列を取得する関数
    
    Args:
        adjacency_matrices: create_normalized_adjacency_matricesで作成された辞書
        hop: 取得したいhop数（1, 2, 3, 4, ...）
    
    Returns:
        torch.Tensor: 指定されたhop数の隣接行列（スパーステンソル）
    """
    key = f'adj_{hop}hop'
    if key in adjacency_matrices:
        return adjacency_matrices[key]
    else:
        raise KeyError(f"Hop {hop}の隣接行列が見つかりません。利用可能なhop: {list(adjacency_matrices.keys())}")
