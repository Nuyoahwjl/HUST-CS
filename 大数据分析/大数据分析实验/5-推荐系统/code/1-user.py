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
output_path = os.path.join(project_root, 'output/1-user')


def load_data():
    """加载所有必要数据"""
    movies = pd.read_csv(os.path.join(input_path, 'movies.csv'))
    train = pd.read_csv(os.path.join(input_path, 'train_set.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test_set.csv'))
    return movies, train, test


def build_structures(train_data):
    """构建数据结构"""
    # 用户评分数据 {user_id: {movie_id: rating}}
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


def normalize_ratings(user_ratings, user_avg):
    """对用户评分进行归一化（减去用户平均评分）"""
    normalized_ratings = defaultdict(dict) 
    for user, ratings in user_ratings.items():
        avg = user_avg[user]
        for movie, rating in ratings.items():
            normalized_ratings[user][movie] = rating - avg
    return normalized_ratings


def cosine_similarity(user1, user2, normalized_ratings):
    """计算两个用户的余弦相似度"""
    common_movies = set(normalized_ratings[user1]) & set(normalized_ratings[user2])
    if not common_movies:
        return 0.0
    
    numerator = sum(normalized_ratings[user1][mid] * normalized_ratings[user2][mid] for mid in common_movies)
    denominator1 = math.sqrt(sum(rating ** 2 for rating in normalized_ratings[user1].values()))
    denominator2 = math.sqrt(sum(rating ** 2 for rating in normalized_ratings[user2].values()))
    
    if denominator1 == 0 or denominator2 == 0:
        return 0.0
    
    return numerator / (denominator1 * denominator2)


def compute_all_similarities(user_ratings, user_avg):
    """预计算所有用户相似度（基于余弦相似度）"""
    # 对评分进行归一化
    normalized_ratings = normalize_ratings(user_ratings, user_avg)
    users = list(normalized_ratings.keys())
    sim_matrix = defaultdict(list)
    
    for i, u in enumerate(tqdm(users, desc="")):
        for v in users[i+1:]:
            sim = cosine_similarity(u, v, normalized_ratings)
            # if sim > 0:
            sim_matrix[u].append((v, sim))
            sim_matrix[v].append((u, sim))
        # 按相似度降序排序
        sim_matrix[u].sort(key=lambda x: x[1], reverse=True)
    
    return sim_matrix


def predict_rating(user_id, movie_id, user_ratings, user_avg, sim_matrix, k=30, global_avg=3.0):
    """预测用户对电影的评分"""
    # 冷启动处理
    if user_id not in user_ratings:
        return global_avg  # 默认评分为全局平均分
    
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
    user_ratings, user_avg = build_structures(train)
    
    # 计算相似度矩阵
    print("Computing similarity matrix...")
    sim_matrix = compute_all_similarities(user_ratings, user_avg)
    
    # 处理测试集
    print("Processing test set...")
    sse = 0.0
    predictions = []
    
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
    print("Results saved to output/1-user.")

    # 为指定用户推荐电影
    user_id = int(input("Enter user ID: "))
    k = int(input("Enter the number of similar users to consider: "))
    n = int(input("Enter the number of movies to recommend: "))
    # 获取用户未评分的电影
    rated_movies = set(user_ratings[user_id].keys())
    all_movies = set(movies['movieId'])
    unrated_movies = all_movies - rated_movies
    # 对未评分电影进行评分预测
    movie_scores = []
    for movie_id in unrated_movies:
        predicted_rating = predict_rating(user_id, movie_id, user_ratings, user_avg, sim_matrix, k=k)
        movie_scores.append((movie_id, predicted_rating))
    # 按评分降序排序并选取前n部电影
    top_n_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:n]
    # 输出推荐结果
    print(f"Top {n} recommended movies for user {user_id}:")
    for movie_id, predicted_rating in top_n_movies:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(f"Movie ID: {movie_id}, Title: {title}, Predicted Rating: {predicted_rating:.2f}")


if __name__ == "__main__":
    main()