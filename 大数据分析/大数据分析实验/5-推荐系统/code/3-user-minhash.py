import math
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm 

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 输入文件路径
input_path = os.path.join(project_root, 'data')
# 输出文件路径
output_path = os.path.join(project_root, 'output/3-user-minhash')

def load_data():
    """加载所有必要数据"""
    movies = pd.read_csv(os.path.join(input_path, 'movies.csv'))
    train = pd.read_csv(os.path.join(input_path, 'train_set.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test_set.csv'))
    return movies, train, test

def build_structures(train_data):
    """构建数据结构"""
    # 用户原始评分数据 {user_id: {movie_id: rating}}
    user_ratings = defaultdict(dict)
    # 用户二值化评分数据 {user_id: {movie_id: 0或1}}
    binary_user_ratings = defaultdict(dict)
    # 用户平均评分 {user_id: avg_rating}
    user_avg = {}
    
    for _, row in train_data.iterrows():
        uid = row['userId']
        mid = row['movieId']
        rating = row['rating']
        user_ratings[uid][mid] = rating
        # 二值化处理：0.5-2.5为0，3.0-5.0为1
        binary_rating = 1 if rating >= 3.0 else 0
        binary_user_ratings[uid][mid] = binary_rating
    
    # 计算用户平均评分
    for uid, ratings in user_ratings.items():
        user_avg[uid] = np.mean(list(ratings.values()))
    
    return user_ratings, binary_user_ratings, user_avg

def generate_minhash_signatures(binary_user_ratings, num_hashes=100, p=2**31-1):
    """生成MinHash签名矩阵"""
    np.random.seed(42)  # 固定随机种子确保可重复性
    a_params = np.random.randint(1, p, size=num_hashes)
    b_params = np.random.randint(0, p, size=num_hashes)
    signatures = {}
    
    for user in tqdm(binary_user_ratings, desc=""):
        # 获取用户评分为1的电影集合
        movies = [mid for mid, rating in binary_user_ratings[user].items() if rating == 1]
        user_signature = []
        
        for i in range(num_hashes):
            a = a_params[i]
            b = b_params[i]
            if not movies:
                # 处理空集合的情况（用户没有评分为1的电影）
                min_hash = p
            else:
                # 计算当前哈希函数的最小哈希值
                hash_values = [(a * mid + b) % p for mid in movies]
                min_hash = min(hash_values)
            user_signature.append(min_hash)
        signatures[user] = user_signature
    
    return signatures

def compute_similarity_matrix(signatures):
    """计算基于MinHash的相似度矩阵"""
    sim_matrix = defaultdict(list)
    users = list(signatures.keys())
    if not users:
        return sim_matrix
    num_hashes = len(signatures[users[0]])
    
    for i in tqdm(range(len(users)), desc=""):
        u = users[i]
        sig_u = signatures[u]
        for j in range(i + 1, len(users)):
            v = users[j]
            sig_v = signatures[v]
            # 计算Jaccard相似度估计值
            matches = sum(s1 == s2 for s1, s2 in zip(sig_u, sig_v))
            similarity = matches / num_hashes
            if similarity > 0:
                sim_matrix[u].append((v, similarity))
                sim_matrix[v].append((u, similarity))
        # 按相似度降序排序
        sim_matrix[u].sort(key=lambda x: x[1], reverse=True)
    
    return sim_matrix

def predict_rating(user_id, movie_id, user_ratings, user_avg, sim_matrix, k=30, global_avg=3.0):
    """预测用户对电影的评分"""
    # 冷启动处理
    if user_id not in user_ratings:
        return global_avg
    
    # 获取top k相似用户
    neighbors = sim_matrix.get(user_id, [])[:k]
    numerator = 0.0
    denominator = 0.0
    avg_u = user_avg[user_id]
    
    for (neighbor, sim) in neighbors:
        # 邻居用户是否评价过该电影
        if movie_id in user_ratings[neighbor]:
            avg_v = user_avg[neighbor]
            rating_v = user_ratings[neighbor][movie_id]
            numerator += sim * (rating_v - avg_v)
            denominator += abs(sim)
    
    if denominator == 0:
        # 没有可用邻居时返回用户平均分
        return max(0.5, min(avg_u, 5.0))
    
    predicted = avg_u + (numerator / denominator)
    return max(0.5, min(predicted, 5.0))  # 截断到[0.5, 5.0]范围

def main():
    # 加载数据
    movies, train, test = load_data()

    # 构建数据结构
    user_ratings, binary_user_ratings, user_avg = build_structures(train)
    
    # 生成MinHash签名
    print("Creating MinHash signatures...")
    signatures = generate_minhash_signatures(binary_user_ratings)
    
    # 计算相似度矩阵
    print("Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(signatures)
    
    # 处理测试集
    print("Processing test set...")
    sse = 0.0
    predictions = []
    
    # 计算全局平均分
    global_avg = np.mean([rating for u_ratings in user_ratings.values() for rating in u_ratings.values()])
    
    for _, row in tqdm(test.iterrows(), total=len(test), desc=""):
        uid = row['userId']
        mid = row['movieId']
        true_rating = row['rating']
        pred = predict_rating(uid, mid, user_ratings, user_avg, sim_matrix, global_avg=global_avg)
        predictions.append(pred)
        sse += (pred - true_rating) ** 2
    
    # 保存结果
    result_df = test.copy()
    result_df['predicted'] = predictions
    os.makedirs(output_path, exist_ok=True)
    
    result_df.to_csv(os.path.join(output_path, 'predictions.csv'), index=False)
    with open(os.path.join(output_path, 'sse.txt'), 'w') as f:
        f.write(f"SSE: {sse}\n")
    
    print(f"SSE: {sse:.4f}")
    print("Results saved to output/3-user-minhash.")

    # 推荐电影功能
    user_id = int(input("Enter user ID: "))
    k = int(input("Enter the number of similar users to consider: "))
    n = int(input("Enter the number of movies to recommend: "))
    
    # 获取用户已评分的电影
    rated_movies = set(user_ratings[user_id].keys()) if user_id in user_ratings else set()
    # 获取所有电影
    all_movies = set(movies['movieId'])
    # 计算未评分的电影
    unrated_movies = all_movies - rated_movies
    
    # 预测评分
    movie_scores = []
    for movie_id in unrated_movies:
        predicted_rating = predict_rating(user_id, movie_id, user_ratings, user_avg, sim_matrix, k=k)
        movie_scores.append((movie_id, predicted_rating))
    
    # 按预测评分排序
    top_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:n]
    
    # 输出结果
    print(f"Top {n} recommended movies for user {user_id}:")
    for movie_id, score in top_movies:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(f"Movie ID: {movie_id}, Title: {title}, Predicted Rating: {predicted_rating:.2f}")

if __name__ == "__main__":
    main()