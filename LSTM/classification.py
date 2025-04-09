import pandas as pd
import sqlite3

# 업로드된 CSV 파일 경로
file_path = './csic_database.csv'

# 데이터 로드
data = pd.read_csv(file_path)

# 정상 데이터와 비정상 데이터 분리
normal_data = data[data['classification'] == 0]
abnormal_data = data[data['classification'] == 1]

# SQLite DB 파일로 저장하는 함수
def save_to_db(df, db_path, table_name):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

# DB 파일 경로 설정
normal_db_path = './normal_data.db'
abnormal_db_path = './abnormal_data.db'

# 정상 데이터와 비정상 데이터 각각 DB에 저장
save_to_db(normal_data, normal_db_path, 'normal_data')
save_to_db(abnormal_data, abnormal_db_path, 'abnormal_data')

# 저장 완료 메시지
print(f"정상 데이터를 {normal_db_path}에 저장했습니다.")
print(f"비정상 데이터를 {abnormal_db_path}에 저장했습니다.")
