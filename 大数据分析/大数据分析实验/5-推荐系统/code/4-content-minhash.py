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
output_path = os.path.join(project_root, 'output/4-content-minhash')

def load_data():
    """加载所有必要数据"""
    movies = pd.read_csv(os.path.join(input_path, 'movies.csv'))
    train = pd.read_csv(os.path.join(input_path, 'train_set.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test_set.csv'))
    return movies, train, test

def build_structures(train_data):
    """构建用户评分数据结构"""
    # 用户原始评分数据 {user_id: {movie_id: rating}}
    user_ratings = defaultdict(dict)
    # 用户平均评分 {user_id: avg_rating}
    user_avg = {}
    
    for _, row in train_data.iterrows():
        uid = row['userId']
        mid = row['movieId']
        rating = row['rating']
        user_ratings[uid][mid] = rating
    
    # 计算用户平均评分
    for uid, ratings in user_ratings.items():
        user_avg[uid] = np.mean(list(ratings.values()))
    
    return user_ratings, user_avg

def prepare_movie_features(movies):
    """构建电影特征矩阵"""
    # 提取所有电影类型
    genres = set()
    for genre_list in movies['genres'].dropna().str.split('|'):
        genres.update(genre_list)
    genres = sorted(genres)
    
    # 创建电影特征字典 {movie_id: set(feature_indices)}
    movie_features = defaultdict(set)
    for _, row in movies.iterrows():
        mid = row['movieId']
        if pd.isna(row['genres']):
            continue
        for genre in row['genres'].split('|'):
            if genre in genres:
                idx = genres.index(genre)
                movie_features[mid].add(idx)
    return movie_features, genres

def generate_minhash_signatures(movie_features, num_hashes=100, p=2**31-1):
    """生成电影MinHash签名"""
    np.random.seed(42)
    a_params = np.random.randint(1, 2**15, size=num_hashes)
    b_params = np.random.randint(0, 2**15, size=num_hashes)
    signatures = {}
    
    for mid in tqdm(movie_features, desc=""):
        features = movie_features[mid]
        signature = []
        for i in range(num_hashes):
            a = a_params[i]
            b = b_params[i]
            if not features:
                min_hash = p
            else:
                hash_values = [(a * f + b) % p for f in features]
                min_hash = min(hash_values)
            signature.append(min_hash)
        signatures[mid] = signature
    return signatures

def compute_similarity_matrix(signatures):
    """计算电影相似度矩阵"""
    sim_matrix = defaultdict(list)
    movies = list(signatures.keys())
    if not movies:
        return sim_matrix
    num_hashes = len(signatures[movies[0]])
    
    for i in tqdm(range(len(movies)), desc=""):
        m1 = movies[i]
        sig_m1 = signatures[m1]
        for j in range(i+1, len(movies)):
            m2 = movies[j]
            sig_m2 = signatures[m2]
            matches = sum(s1 == s2 for s1, s2 in zip(sig_m1, sig_m2))
            similarity = matches / num_hashes
            if similarity > 0:
                sim_matrix[m1].append((m2, similarity))
                sim_matrix[m2].append((m1, similarity))
        # 按相似度降序排列
        sim_matrix[m1].sort(key=lambda x: x[1], reverse=True)
    return sim_matrix

def predict_rating(user_id, movie_id, user_ratings, user_avg, movie_sim_matrix, global_avg=3.0):
    """基于内容预测评分（考虑所有评过分的电影）"""
    # 冷启动处理
    if user_id not in user_ratings:
        return global_avg
    
    user_rated = user_ratings[user_id]
    avg_rating = user_avg.get(user_id, global_avg)
    
    # 获取所有相似电影中用户评分过的
    neighbors = []
    for (similar_movie, similarity) in movie_sim_matrix.get(movie_id, []):
        if similar_movie in user_rated:
            neighbors.append((similar_movie, similarity))
    
    if not neighbors:
        return avg_rating
    
    # 计算加权平均
    numerator = sum(user_rated[m] * sim for m, sim in neighbors)
    denominator = sum(sim for _, sim in neighbors)
    
    if denominator == 0:
        return avg_rating
    
    predicted = numerator / denominator
    return max(0.5, min(predicted, 5.0))  # 截断到[0.5, 5.0]

def main():
    # 加载数据
    movies, train, test = load_data()
    
    # 构建用户评分数据
    user_ratings, user_avg = build_structures(train)
    
    # 准备电影特征
    print("Preparing movie features...")
    movie_features, genres = prepare_movie_features(movies)
    
    # 生成MinHash签名
    print("Generating MinHash signatures...")
    signatures = generate_minhash_signatures(movie_features)
    
    # 计算相似度矩阵
    print("Computing similarity matrix...")
    movie_sim_matrix = compute_similarity_matrix(signatures)
    
    # 处理测试集
    print("Processing test set...")
    sse = 0.0
    predictions = []
    global_avg = np.mean([r for u in user_ratings.values() for r in u.values()])
    
    for _, row in tqdm(test.iterrows(), total=len(test)):
        uid = row['userId']
        mid = row['movieId']
        true_rating = row['rating']
        pred = predict_rating(uid, mid, user_ratings, user_avg, movie_sim_matrix, global_avg=global_avg)
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
    print("Results saved to output/4-content-minhash.")

    # 推荐功能
    user_id = int(input("Enter user ID: "))
    k = int(input("Number of recommendations: "))
    
    # 获取用户未评分的电影
    rated_movies = set(user_ratings.get(user_id, {}).keys())
    all_movies = set(movies['movieId'])
    unrated_movies = all_movies - rated_movies
    
    # 预测未评分电影的评分
    recommendations = []
    for mid in unrated_movies:
        pred = predict_rating(user_id, mid, user_ratings, user_avg, movie_sim_matrix)
        recommendations.append((mid, pred))
    
    # 获取top k推荐
    top_movies = sorted(recommendations, key=lambda x: x[1], reverse=True)[:k]
    
    # 输出结果
    print(f"\nTop {k} recommendations for user {user_id}:")
    for mid, score in top_movies:
        title = movies[movies['movieId'] == mid]['title'].values[0]
        print(f"Movie ID: {mid}, Title: {title}, Predicted Rating: {score:.2f}")

if __name__ == "__main__":
    main()