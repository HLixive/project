import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip
import speech_recognition as sr
import cx_Oracle


# Oracle 데이터베이스 연결 정보
db_username = "LIV 접속"
db_password = "1226"
db_host = "localhost"
db_port = "1521"
db_service_name = "orcl"

def select_files():
    file_paths = filedialog.askopenfilenames(title="파일 선택")
    if file_paths:
        file_entry.delete("1.0", tk.END)
        file_entry.insert(tk.END, "".join(file_paths))


def find_word_in_file(file_path, target_word):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            word_count = content.count(word)
            return word_count
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return 0

def classify_files():
    # 파일 경로 목록을 가져오는 코드 작성
    file_paths = ['파일1.txt', '파일2.txt', '파일3.txt']
    
    # 수정된 부분: 단일 파일 경로를 사용하므로 반복문 삭제
    for file_path in file_paths:
        count = find_word_in_file(file_path, target_word)
        print(f"'{target_word}' 단어는 파일 '{file_path}'에서 {count}번 등장합니다.")


    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1]

        try:
            if file_extension.lower() in ['.mp3', '.wav','.m4a']:
                text = transcribe_audio(file_path)
            elif file_extension.lower() in ['.mp4', '.avi']:
                text = transcribe_video(file_path)
            else:
                messagebox.showwarning("경고", f"지원하지 않는 파일 형식입니다: {file_extension}")
                continue

            if keyword.lower() in text.lower():
                result_label.config(text=f"'{file_name}' 파일은 해당 단어를 포함합니다.")
                save_result(file_name, keyword, True)
            else:
                result_label.config(text=f"'{file_name}' 파일은 해당 단어를 포함하지 않습니다.")
                save_result(file_name, keyword, False)
        except Exception as e:
            messagebox.showerror("오류", f"파일 처리 중 오류가 발생했습니다: {str(e)}")

def transcribe_video(file_path):
    video = VideoFileClip(file_path)
    audio = video.audio

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio.filename) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language='ko-KR')

    return text


def transcribe_audio(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.m4a':
        audio = AudioSegment.from_file(file_path, format='.m4a')
        audio.export("temp.wav", format=".wav")
    else:
        audio = AudioSegment.from_file(file_path)
        audio.export("temp.wav", format=".wav")

    r = sr.Recognizer()
    with sr.AudioFile("temp.wav") as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language="ko-KR")

    os.remove("temp.wav")
    return text


def save_result(file_name, keyword, result):
    conn = cx_Oracle.connect(f"{db_username}/{db_password}@{db_host}:{db_port}/{db_service_name}")
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE file_results
                      (id NUMBER PRIMARY KEY,
                      file_name VARCHAR2(100),
                      keyword VARCHAR2(100),
                      result NUMBER)''')

    cursor.execute('INSERT INTO file_results (file_name, keyword, result) VALUES (:file_name, :keyword, :result)',
                   file_name=file_name, keyword=keyword, result=int(result))

    conn.commit()
    conn.close()

root = tk.Tk()
root.title("음성/영상 단어 찾기")
root.geometry("400x250")

keyword_label = tk.Label(root, text="찾을 단어:")
keyword_label.place(x=20, y=20)

keyword_entry = tk.Entry(root)
keyword_entry.place(x=120, y=20)

file_label = tk.Label(root, text="파일 선택:")
file_label.place(x=20, y=60)

file_entry = tk.Text(root, height=5, width=30)
file_entry.place(x=120, y=60)

file_button = tk.Button(root, text="파일 선택", command=select_files)
file_button.place(x=220, y=140)

classify_button = tk.Button(root, text="단어 찾기", command=classify_files)
classify_button.place(x=120, y=140)

result_label = tk.Label(root, text="")
result_label.place(x=20, y=180)

root.mainloop()
