import sqlite3
import re
from datetime import datetime

# 로그 파일 경로
log_file_path = r'D:\Pay1oad 프로젝트\.venv\access.log'

# SQLite DB 생성 및 연결
conn = sqlite3.connect('logs.db')
cur = conn.cursor()

# 테이블 생성 (처음 한 번만)
cur.execute('''
CREATE TABLE IF NOT EXISTS nginx_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip TEXT,
    timestamp DATETIME,
    method TEXT,
    url TEXT,
    protocol TEXT,
    status_code INTEGER,
    response_size INTEGER,
    referrer TEXT,
    user_agent TEXT
)
''')
conn.commit()

# 로그 파싱 & DB에 저장
with open(log_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        pattern = r'(\S+) - - \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+) "(.*?)" "(.*?)"'
        match = re.match(pattern, line)
        if match:
            try:
                ip = match.group(1)
                dt = datetime.strptime(match.group(2).split()[0], "%d/%b/%Y:%H:%M:%S")
                method = match.group(3)
                url = match.group(4)
                protocol = match.group(5)
                status_code = int(match.group(6))
                response_size = int(match.group(7))
                referrer = match.group(8)
                user_agent = match.group(9)

                cur.execute('''
                INSERT INTO nginx_logs (ip, timestamp, method, url, protocol, status_code, response_size, referrer, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (ip, dt, method, url, protocol, status_code, response_size, referrer, user_agent))
            except Exception as e:
                print(f"[SKIP] 에러 발생한 라인: {line.strip()} \n이유: {e}")

conn.commit()
conn.close()
print("✅ 로그를 logs.db로 성공적으로 변환했습니다!")
