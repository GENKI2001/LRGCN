"""
グラフの各hopにおけるクラス分布ベクトルを計算し、ヒートマップで可視化するスクリプト
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_undirected, add_self_loops, degree
import argparse

from utils.dataset import load_dataset
from utils.graph import make_undirected


def create_khop_adjacency_matrices(data, device, max_hops=4):
    """
    k-hopまでの隣接行列を作成（exactly k-hopのみ）
    
    Args:
        data: PyTorch Geometricのデータオブジェクト
        device: 計算デバイス
        max_hops: 最大hop数
    
    Returns:
        dict: {f'adj_{k}hop': sparse_tensor} の辞書
    """
    out = {}
    
    # 1-hop（無向・自己ループあり）
    edge_index_undirected = to_undirected(data.edge_index, num_nodes=data.num_nodes).to(device)
    edge_index_with_loops, _ = add_self_loops(edge_index_undirected, num_nodes=data.num_nodes)
    edge_index_with_loops = edge_index_with_loops.to(device)
    row, col = edge_index_with_loops
    
    # 1-hopのbool隣接行列（dense）
    A1_bool = torch.sparse_coo_tensor(
        indices=edge_index_with_loops.to(device),
        values=torch.ones(edge_index_with_loops.size(1), device=device),
        size=(data.num_nodes, data.num_nodes),
        device=device,
    ).to_dense().bool()
    
    out['adj_1hop'] = A1_bool
    
    # 既知hopの集合（論理和）
    reached_bool = A1_bool.clone()
    
    # A_k_boolを順に構築
    Ak_bool = A1_bool
    for k in range(2, max_hops + 1):
        # Ak = A_{k-1} * A1
        Ak_bool = (Ak_bool.float().mm(A1_bool.float()) > 0)
        
        # exactly k-hop = Ak_boolから1..(k-1)hopを除去
        exact_k_bool = Ak_bool & (~reached_bool)
        
        out[f'adj_{k}hop'] = exact_k_bool
        
        # 既知集合を更新
        reached_bool |= exact_k_bool
    
    return out


def compute_class_distribution_vectors(data, khop_adj_matrices, device, max_hops=4):
    """
    各hopにおけるクラス分布ベクトルを計算
    
    Args:
        data: PyTorch Geometricのデータオブジェクト
        khop_adj_matrices: create_khop_adjacency_matricesで作成された辞書
        device: 計算デバイス
        max_hops: 最大hop数
    
    Returns:
        dict: {hop: [num_classes, num_classes]} の辞書
            - 各行はノードクラス
            - 各列はネイバークラス
            - 値は確率（0-1）
    """
    num_classes = int(data.y.max().item() + 1)
    num_nodes = data.num_nodes
    
    # 各ノードのクラスを取得
    node_classes = data.y.cpu().numpy()
    
    # 結果を格納する辞書
    class_distributions = {}
    
    for hop in range(1, max_hops + 1):
        adj_key = f'adj_{hop}hop'
        if adj_key not in khop_adj_matrices:
            print(f"警告: {adj_key}が見つかりません")
            continue
        
        # k-hop隣接行列（bool）
        khop_adj = khop_adj_matrices[adj_key]  # [num_nodes, num_nodes]
        
        # クラス分布ベクトルを格納する行列 [num_classes, num_classes]
        distribution_matrix = np.zeros((num_classes, num_classes))
        
        # 各ノードクラスについて
        for node_class in range(num_classes):
            # このクラスに属するノードのインデックス
            nodes_in_class = np.where(node_classes == node_class)[0]
            
            if len(nodes_in_class) == 0:
                continue
            
            # これらのノードのk-hopネイバーのクラスを集計
            neighbor_class_counts = np.zeros(num_classes)
            
            for node_idx in nodes_in_class:
                # このノードのk-hopネイバーを取得
                neighbor_mask = khop_adj[node_idx].cpu().numpy()  # [num_nodes] bool array
                neighbors = np.where(neighbor_mask)[0]
                
                if len(neighbors) == 0:
                    continue
                
                # ネイバーのクラスを集計
                neighbor_classes = node_classes[neighbors]
                for nc in neighbor_classes:
                    neighbor_class_counts[int(nc)] += 1
            
            # 確率分布に正規化
            total_neighbors = neighbor_class_counts.sum()
            if total_neighbors > 0:
                distribution_matrix[node_class] = neighbor_class_counts / total_neighbors
            else:
                # ネイバーがいない場合は均一分布
                distribution_matrix[node_class] = 1.0 / num_classes
        
        class_distributions[hop] = distribution_matrix
    
    return class_distributions


def plot_class_distribution_heatmaps(class_distributions, dataset_name, output_path=None):
    """
    クラス分布ベクトルをヒートマップで可視化
    
    Args:
        class_distributions: compute_class_distribution_vectorsの結果
        dataset_name: データセット名
        output_path: 出力ファイルパス（Noneの場合は表示のみ）
    """
    num_hops = len(class_distributions)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'{dataset_name} Dataset: Class Distribution Vectors Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 各hopのカラースキームを設定
    color_schemes = ['Blues', 'Reds', 'Greens', 'Purples']
    
    for idx, (hop, dist_matrix) in enumerate(sorted(class_distributions.items())):
        ax = axes[idx]
        
        # ヒートマップを描画
        num_classes = dist_matrix.shape[0]
        sns.heatmap(
            dist_matrix,
            ax=ax,
            cmap=color_schemes[idx],
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Probability'},
            xticklabels=[f'Class {i}' for i in range(num_classes)],
            yticklabels=[f'Class {i}' for i in range(num_classes)],
            vmin=0.0,
            vmax=1.0 if hop == 1 else None,  # 1-hop以外は自動スケール
        )
        
        ax.set_title(f'{hop}-Hop Class Distribution Vectors', fontsize=12, fontweight='bold')
        ax.set_xlabel('Neighbor Class', fontsize=10)
        ax.set_ylabel('Node Class', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze class distribution vectors at different hops')
    parser.add_argument('--dataset_name', type=str, default='Texas', 
                       help='Dataset name (e.g., Texas, Cornell, Wisconsin, Cora, Citeseer, Pubmed)')
    parser.add_argument('--max_hops', type=int, default=4, help='Maximum number of hops to analyze')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output file path for the figure (e.g., class_distribution_texas.png)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセットの読み込み
    print(f"Loading dataset: {args.dataset_name}")
    dataset, canonical_name = load_dataset(args.dataset_name)
    data = dataset[0].clone()
    
    print(f"Dataset: {canonical_name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Number of classes: {int(data.y.max().item() + 1)}")
    
    # make_undirectedを適用
    print("Applying make_undirected...")
    data = make_undirected(data, device)
    data = data.to(device)
    
    # k-hop隣接行列を作成
    print(f"Creating {args.max_hops}-hop adjacency matrices...")
    khop_adj_matrices = create_khop_adjacency_matrices(data, device, max_hops=args.max_hops)
    
    # クラス分布ベクトルを計算
    print("Computing class distribution vectors...")
    class_distributions = compute_class_distribution_vectors(
        data, khop_adj_matrices, device, max_hops=args.max_hops
    )
    
    # 結果を表示
    print("\nClass Distribution Vectors:")
    print("=" * 60)
    for hop in sorted(class_distributions.keys()):
        print(f"\n{hop}-Hop Distribution:")
        dist_matrix = class_distributions[hop]
        for i in range(dist_matrix.shape[0]):
            print(f"  Class {i}: {dist_matrix[i]}")
    
    # ヒートマップで可視化
    print("\nGenerating heatmaps...")
    output_path = args.output or f"class_distribution_{canonical_name.lower()}.png"
    plot_class_distribution_heatmaps(class_distributions, canonical_name, output_path)
    
    print("\n完了しました！")


if __name__ == "__main__":
    main()

