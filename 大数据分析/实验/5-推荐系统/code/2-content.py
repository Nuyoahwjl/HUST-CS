import math
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm


# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
input_path = os.path.join(project_root, 'data')
output_path = os.path.join(project_root, 'output/2-content')


def load_data():
    """加载所有必要数据"""
    movies = pd.read_csv(os.path.join(input_path, 'movies.csv'))
    train = pd.read_csv(os.path.join(input_path, 'train_set.csv'))
    test = pd.read_csv(os.path.join(input_path, 'test_set.csv'))
    return movies, train, test


def build_structures(train_data):
    """构建用户评分数据结构"""
    user_ratings = defaultdict(dict)
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


def compute_movie_similarity(movies):
    """实现电影相似度计算"""
    # 处理genres字段
    movies_genres = movies['genres'].apply(lambda x: x.split('|')).tolist()
    
    # 统计文档频率
    doc_freq = defaultdict(int)
    for genres in movies_genres:
        unique_words = set(genres)
        for word in unique_words:
            doc_freq[word] += 1
    
    N = len(movies_genres)  # 总文档数
    
    # 计算IDF（使用平滑处理）
    idf_dict = {}
    for word, df in doc_freq.items():
        idf = math.log((N + 1) / (df + 1)) + 1
        idf_dict[word] = idf
    
    # 构建词汇表
    vocab = list(doc_freq.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    
    # 生成TF-IDF矩阵
    tfidf_matrix = []
    print("Computing TF-IDF matrix...")
    for genres in tqdm(movies_genres, desc=""):
        total_words = len(genres)
        word_counts = defaultdict(int)
        for word in genres:
            word_counts[word] += 1
        
        vec = [0.0] * vocab_size
        if total_words == 0:
            tfidf_matrix.append(vec)
            continue
        
        for word, count in word_counts.items():
            idx = word_to_idx.get(word, -1)
            if idx == -1:
                continue  
            tf = count / total_words
            vec[idx] = tf * idf_dict[word]
        tfidf_matrix.append(vec)
    
    # 计算余弦相似度矩阵
    movie_sim_matrix = compute_cosine_similarity(tfidf_matrix)
    
    # 创建movieId到索引的映射
    movie_ids = movies['movieId'].tolist()
    movie_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    return movie_sim_matrix, movie_to_index


def compute_cosine_similarity(tfidf_matrix):
    """计算余弦相似度矩阵"""
    n = len(tfidf_matrix)
    # 预计算每个向量的模长
    norms = [math.sqrt(sum(x**2 for x in vec)) for vec in tfidf_matrix]
    sim_matrix = []
    print("Computing cosine similarity matrix...")
    for i in tqdm(range(n), desc=""):
        row = []
        vec_i = tfidf_matrix[i]
        norm_i = norms[i]
        for j in range(n):
            vec_j = tfidf_matrix[j]
            norm_j = norms[j]
            # 计算点积
            dot_product = sum(a * b for a, b in zip(vec_i, vec_j))
            # 处理零模长的情况
            if norm_i == 0 or norm_j == 0:
                sim = 0.0
            else:
                sim = dot_product / (norm_i * norm_j)
            row.append(sim)
        sim_matrix.append(row)
    return sim_matrix


def predict_rating(user_id, movie_id, user_ratings, user_avg, movie_sim_matrix, movie_to_index, global_avg):
    """基于电影相似度的评分预测"""
    # 处理冷启动用户
    if user_id not in user_ratings:
        return global_avg
    
    # 处理未知电影
    if movie_id not in movie_to_index:
        return user_avg.get(user_id, global_avg)
    
    target_idx = movie_to_index[movie_id]
    sum_sim = 0.0
    sum_rating_sim = 0.0
    
    # 遍历用户评分过的所有电影
    for rated_mid, rating in user_ratings[user_id].items():
        if rated_mid not in movie_to_index:
            continue
        
        rated_idx = movie_to_index[rated_mid]
        sim = movie_sim_matrix[rated_idx][target_idx]
        
        if sim > 0:
            sum_sim += sim
            sum_rating_sim += rating * sim
    
    if sum_sim > 0:
        predicted = sum_rating_sim / sum_sim
    else:
        # 没有相似电影时使用用户平均分
        predicted = user_avg.get(user_id, global_avg)
    
    return max(0.5, min(predicted, 5.0))  # 评分截断

def main():
    # 加载数据
    movies, train, test = load_data()
    
    # 构建用户评分数据
    user_ratings, user_avg = build_structures(train)
    global_avg = train['rating'].mean()
    
    # 计算电影相似度矩阵
    movie_sim_matrix, movie_to_index = compute_movie_similarity(movies)
    
    # 处理测试集
    print("Processing test set...")
    sse = 0.0
    predictions = []
    
    for _, row in tqdm(test.iterrows(), total=len(test), desc=""):
        uid = row['userId']
        mid = row['movieId']
        true_rating = row['rating']
        
        pred = predict_rating(uid, mid, user_ratings, user_avg, 
                             movie_sim_matrix, movie_to_index, global_avg)
        
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
    print("Results saved to output/2-content.")

    # 推荐功能
    user_id = int(input("Enter user ID: "))
    k = int(input("Enter number of recommendations: "))
    
    # 获取所有电影ID
    all_movie_ids = movies['movieId'].unique()
    # 获取用户已评分的电影
    rated_movies = user_ratings.get(user_id, {}).keys()
    # 生成未评分电影列表
    unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movies]
    
    # 预测评分
    predictions = []
    print("Predicting ratings for unrated movies...")
    for mid in tqdm(unrated_movies, desc=""):
        pred = predict_rating(user_id, mid, user_ratings, user_avg,
                             movie_sim_matrix, movie_to_index, global_avg)
        predictions.append((mid, pred))
    
    # 按评分排序取前k个
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:k]
    
    # 输出结果
    print(f"Top {k} recommendations for user {user_id}:")
    for mid, score in top_movies:
        title = movies[movies['movieId'] == mid]['title'].values[0]
        print(f"Movie ID: {mid}, Title: {title}, Predicted Rating: {score:.2f}")


if __name__ == "__main__":
    main()



# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# def compute_movie_similarity(movies):
#     """计算电影相似度矩阵"""
#     # 创建TF-IDF矩阵
#     vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None)
#     tfidf_matrix = vectorizer.fit_transform(movies['genres'])
#     # 计算余弦相似度
#     movie_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     # 创建movieId到索引的映射
#     movie_ids = movies['movieId'].tolist()
#     movie_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
#     return movie_sim_matrix, movie_to_index