import os
from dotenv import load_dotenv

# .env を読み込む
load_dotenv()
print(os.getenv("WHISPER_API"))