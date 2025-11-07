import sys
import json
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from umap import UMAP
from sklearn.decomposition import PCA
import torch
import vllm
from vllm import LLM
from transformers import BertModel, BertTokenizer

class QwenEmbeddingVllm:
    ## Qwen3-embedding 
    def __init__(self, size="Qwen3-Embedding-4B"):
        if size == "Qwen3-Embedding-4B":
            model_path = "/home/share_ssd_data/nfs-data1/muwenjing/models/Qwen3-Embedding-4B"
        elif size == "Qwen3-Embedding-0.6B":
            model_path = "/home/share_ssd_data/nfs-data1/muwenjing/models/Qwen3-Embedding-0.6B"
        elif size == "Qwen3-Embedding-8B":
            model_path = "/home/share_ssd_data/nfs-data1/muwenjing/models/Qwen3-Embedding-8B"
        print(f"加载模型: {model_path}")
        self.model = LLM(model=model_path, task="embed")
    
    def get_detailed_instruct(self, query: str) -> str:
        # Each query must come with a one-sentence instruction that describes the task
        task_description = "给定一个错误原因，找到相关的其他错误原因"  ## 加入效果变差
        task_description = "" 
        return f'Instruct: {task_description}\nQuery:{query}'

    def get_embeddings_batch(self, texts):
        input_texts = [self.get_detailed_instruct(it) for it in texts]
        outputs = self.model.embed(input_texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        return embeddings
    
def select_embedding_algorithm(documents, embed_alg):
    ## 多种embeeding方法选择
    if embed_alg == "use_qwen_0.6B":
        print("选择Qwen3-Embedding-0.6B向量模型")
        qwen_embed = QwenEmbeddingVllm("Qwen3-Embedding-0.6B")
        embeddings = qwen_embed.get_embeddings_batch(documents)
    elif embed_alg == "use_qwen_4B":
        print("选择Qwen3-Embedding-4B向量模型")
        qwen_embed = QwenEmbeddingVllm("Qwen3-Embedding-4B")
        embeddings = qwen_embed.get_embeddings_batch(documents)
    elif embed_alg == "use_qwen_8B":
        print("选择Qwen3-Embedding-8B向量模型")
        qwen_embed = QwenEmbeddingVllm("Qwen3-Embedding-8B")
        embeddings = qwen_embed.get_embeddings_batch(documents)
    return embeddings
  
def reduce_embeddings(embeddings):
        """多种embedding预处理策略"""
        strategies = {}

        strategies['normalized'] = normalize(embeddings, norm='l2')
        # 策略1: PCA降维
        try:
            n_components = min(256, embeddings.shape[0]-1, embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            strategies['pca'] = pca.fit_transform(strategies['normalized'])
            print(f"PCA降维: {embeddings.shape[1]} -> {n_components}维")
            print(f"PCA解释方差: {pca.explained_variance_ratio_.sum():.3f}")
        except Exception as e:
            print(f"PCA失败: {e}")
        '''
        # 策略2: UMAP降维 
        try:
            n_components_umap = min(25, embeddings.shape[0]-1)
            umap_reducer = UMAP(
                n_components=n_components_umap,
                metric='cosine',
                random_state=42,
                n_neighbors=min(15, embeddings.shape[0]-1),
                min_dist=0.1
            )
            strategies['umap'] = umap_reducer.fit_transform(embeddings)
            print(f"UMAP降维到: {strategies['umap'].shape[1]}维")
        except Exception as e:
            print(f"UMAP失败: {e}")
        '''
        
        return strategies

def hdbscan_clustering(documents, embed_alg):
    """聚类算法"""
    ## 多种embeeding方法选择
    embeddings = select_embedding_algorithm(documents, embed_alg)

    #  归一化embedding（提高聚类效果）
    embeddings = normalize(embeddings, norm='l2')
    ### 多种降维策略 
    reduced_embeddings = reduce_embeddings(embeddings)
    embeddings = reduced_embeddings['pca']
 
    # 计算余弦距离矩阵
    similarity_matrix = cosine_similarity(embeddings)   # 计算余弦相似度矩阵
    distance_matrix = 1 - similarity_matrix             # 将相似度转换为距离（距离 = 1 - 相似度）

    # 关键步骤：强制转换为 float64
    distance_matrix = distance_matrix.astype(np.float64)

    # HDBSCAN聚类 - 使用更适合文本的配置
    # min_cluster_size, min_samples = 2, 1
    min_cluster_size, min_samples = 5, 1
    min_cluster_size, min_samples = 10, 1
    # min_cluster_size, min_samples = 2, 2

    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,              # 簇的最小大小
        min_samples=min_samples,                   # 核心点所需的最小邻居数
        metric='precomputed',                 # 对于文本embedding，余弦距离更合适
        cluster_selection_method='eom',  # leaf 相比 eom 可能产生更多簇，之前默认是leaf
        cluster_selection_epsilon=0.1,   # 控制簇的合并
        algorithm='best',                # 使用最佳算法
        alpha=1.0,                        # 控制树状图的紧凑度
    )
    hdbscan_labels = hdbscan_clusterer.fit_predict(distance_matrix)
    
    # 分析结果
    unique_labels = set(hdbscan_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(hdbscan_labels).count(-1)
    
    print("")
    print(f"HDBSCAN结果: {n_clusters}个簇, {n_noise}个噪声点")

    ## 构建embedding和文本对应
    embed_cluster_results = []
    for i, (doc, label, embeding) in enumerate(zip(documents, hdbscan_labels, embeddings)):
        cluster_id = int(label)
        embed_cluster_results.append({"cluster_id":cluster_id, "text":doc, "embedding":embeding.tolist()})

    # 输出详细的聚类结果
    cluster_results = {}
    for i, (doc, label) in enumerate(zip(documents, hdbscan_labels)):
        cluster_id = int(label)
        if cluster_id not in cluster_results:
            cluster_results[cluster_id] = []
        cluster_results[cluster_id].append({
            'text': doc,
            'probability': float(hdbscan_clusterer.probabilities_[i]) if i < len(hdbscan_clusterer.probabilities_) else 0.0
        })
    ## 输出最大的10个簇
    cluster_num_results = {}
    for id, docs in cluster_results.items():
        if id == -1:
            continue
        cluster_num_results[id] = len(docs)
    cluster_num_sorted = sorted(cluster_num_results.items(), key=lambda item:item[1], reverse=True)
    print("输出最大的簇 ....")
    for (cluster_id, num) in cluster_num_sorted[:10]:
        cluster_ratio = float(num)/len(documents)
        print(f"簇 {cluster_id} 共{num}个样本, 占比 {cluster_ratio:.3f}")
    print()

    # 按簇大小排序输出
    for cluster_id in sorted(cluster_results.keys()):
        cluster_docs = cluster_results[cluster_id]
        cluster_type = "噪声点" if cluster_id == -1 else f"簇 {cluster_id}"
        print(f"\n{cluster_type} (共{len(cluster_docs)}个样本):")   
        for item in cluster_docs[:5]:  # 每个簇只显示前5个
            prob_info = f", 概率: {item['probability']:.3f}" if cluster_id != -1 else ""
            print(f"  - {item['text']}{prob_info}")
        if len(cluster_docs) > 5:
            print(f"  ... 还有{len(cluster_docs)-5}个样本")
        if cluster_id == -1:
            continue
        print("按照字符串出现次数排序...")
        doc_num_dict = {}
        for item in cluster_docs:
            text_ = item['text']
            if text_ not in doc_num_dict:
                doc_num_dict[text_] = 0
            doc_num_dict[text_] += 1
        doc_num_sorted = sorted(doc_num_dict.items(), key=lambda item:item[1], reverse=True)
        for (doc, num) in doc_num_sorted:
            print("  - " + doc + ", 次数：" + str(num))

    # 输出结果
    for doc, label in zip(documents, hdbscan_labels):
        noise_info = " (噪声点)" if label == -1 else ""
        print(f"簇 {label}: {doc}{noise_info}")

    return cluster_results, embed_cluster_results

if __name__ == "__main__":
    input_file = sys.argv[1]
    ouput_file = sys.argv[2]
    output_embed_file = sys.argv[3]

    documents = []
    with open(input_file, 'r') as fr:
        for line in fr:
            items = line.strip().split("\t")
            documents.append(items[0])
    documents = list(set(documents))

    embed_alg = "use_qwen_0.6B"
    embed_alg = "use_qwen_8B"
    embed_alg = "use_qwen_4B"
    cluster_results, embed_cluster_results = hdbscan_clustering(documents, embed_alg)

    ## 输出聚类结果
    with open(ouput_file, 'w') as fw:
        for cluster_id in sorted(cluster_results.keys()):
            if cluster_id == -1:
                continue
            cluster_docs = cluster_results[cluster_id]
            cluster_type = "噪声点" if cluster_id == -1 else f"簇 {cluster_id}"
            output_str = f"{cluster_type}\t{len(cluster_docs)}\t"
            doc_num_dict = {}
            for item in cluster_docs:
                text_ = item['text']
                if text_ not in doc_num_dict:
                    doc_num_dict[text_] = 0
                doc_num_dict[text_] += 1
            doc_num_sorted = sorted(doc_num_dict.items(), key=lambda item:item[1], reverse=True)
            for (doc, num) in doc_num_sorted:
                output_str += doc + "(" + str(num)+ ")" + "\t"
            fw.write(output_str + "\n")

    ## 输出embedding结果
    with open(output_embed_file, "w") as fw:
        for res in embed_cluster_results:
            fw.write(json.dumps(res, ensure_ascii=False) + "\n")
    