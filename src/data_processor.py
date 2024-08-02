import re
import os
import json
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import wiener

class DataProcessor:
    def __init__(self, file_path):
        self.data = []
        self.file_path = file_path
        print(DataProcessor.is_audio_file(file_path))
        self.audio, self.sr = sf.read(file_path, dtype='float32') if DataProcessor.is_audio_file(file_path) else [None, None]
        self.processed_audio = self.audio.copy() if DataProcessor.is_audio_file(file_path) else None

    @staticmethod
    def is_audio_file(file_path):
        return any([file_path.endswith(fmt) for fmt in ['.mp3', '.m4a', '.wav']])

    @staticmethod
    def sort_files(filename):
        # ファイル名から数字を取り出してソートの基準とします
        part = filename.split('.')[0]
        try:
            # ファイル名を整数に変換してソート
            return int(part)
        except ValueError:
            # 数字以外のファイル名の場合は無限大を返します
            return float('inf')

    def process_file(self, file_path):
        # ファイルを読み込んでデータに追加します
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data.append(file.read())

    def export_to_txt(self, output_txt):
        # 処理したデータをテキストファイルに出力します
        with open(output_txt, 'w', encoding='utf-8') as file:
            for content in self.data:
                file.write(content + '\n')

    def process_all_files(self, directory_path):
        # 指定されたディレクトリ内のすべての.txtファイルを取得し、ソートして処理します
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        files = sorted(files, key=DataProcessor.sort_files)
        for filename in files:
            file_path = os.path.join(directory_path, filename)
            self.process_file(file_path)

    def calculate_snr(self):
        # 騒音は音声の最初の0.5秒と仮定します
        noise_part = self.audio[0:int(0.5 * self.sr)]
        signal_power = np.mean(self.audio ** 2)
        noise_power = np.mean(noise_part ** 2)
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def apply_noise_reduction_if_needed(self, threshold_snr=10):
        snr = self.calculate_snr()
        if snr < threshold_snr:
            print(f"SNRが{snr} dBのため、ノイズリダクションを適用します")
            # ノイズ除去方法の例としてウィナー フィルタリングを使用する
            self.audio = wiener(self.audio)
        else:
            print(f"SNRが{snr} dBのため、ノイズリダクションは不要です")

    def apply_noise_reduction(self, reduction_method='wiener', intensity=1):
        if reduction_method == 'wiener':
            # Wiener filter for noise reduction
            self.processed_audio = wiener(self.audio, mysize=None, noise=None)
        elif reduction_method == 'noisereduce':
            # NoiseReduce package
            noise_clip = self.audio[0:int(0.5 * self.sr)]  # Assume first 0.5 seconds is noise
            self.processed_audio = nr.reduce_noise(audio_clip=self.audio, noise_clip=noise_clip, verbose=False)

        # Intensity adjustment (not scientifically accurate, just for demonstration)
        self.processed_audio *= intensity

    def adjust_snr(self, target_snr_db):
        # Calculate current SNR
        signal_power = np.mean(self.audio ** 2)
        noise_power = np.mean((self.audio - self.processed_audio) ** 2)
        current_snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Calculate required adjustment factor
        required_snr_linear = 10 ** ((target_snr_db - current_snr_db) / 10)
        self.processed_audio *= required_snr_linear

    def parse_episode(self, output_txt):
        episodes = []
        current_episode = None  # 現在処理中のエピソードを格納する変数
        dialogues = []  # 現在のダイアログを格納するリスト
        episode_start_pattern = re.compile(r'^（(.+)が始まりました）$')
        episode_end_pattern = re.compile(r'^（(.+)は終わりました）$')
        dialogue_pattern = re.compile(r'^说话人(\d+)\s+(\d{2}:\d{2})$')

        # テキストファイルを読み込む
        with open(output_txt, 'r', encoding='utf-8') as file:
            text = file.read()

        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # エピソードの開始を検出する処理
            start_match = episode_start_pattern.match(line)
            if start_match:
                if current_episode is not None:
                    # 現在のエピソードをリストに追加
                    episodes.append(current_episode)
                # 新しいエピソードの開始
                current_episode = {'title': start_match.group(1), 'dialogues': []}
                continue

            # エピソードの終了を検出する処理
            end_match = episode_end_pattern.match(line)
            if end_match:
                # エピソードをリストに追加し、エピソード情報をリセット
                if current_episode is not None:
                    episodes.append(current_episode)
                    current_episode = None
                continue

            # ダイアログの処理
            if current_episode is not None:
                # current_episodeがNoneでない場合にのみ処理を行う
                current_episode['dialogues'].append({
                    'speaker': '発言者',
                    'time': '時間',
                    'text': 'テキスト'
                })

            # ダイアログの解析
            speaker_match = dialogue_pattern.match(line)
            if speaker_match:
                if dialogues:
                    # 前のダイアログをエピソードに追加
                    current_episode['dialogues'].append({
                        'speaker': current_speaker,
                        'time': current_time,
                        'text': ' '.join(dialogues)
                    })
                    dialogues = []
                # 新しいダイアログの開始
                current_speaker = speaker_match.group(1)
                current_time = speaker_match.group(2)
            else:
                # ダイアログの続きを追加
                dialogues.append(line)

        return episodes

    def prompt_engineer(self, output_txt, output_json):
        if len(self.data) != 0:
            self.data = []

        # テキストを解析
        episodes = self.parse_episode(output_txt)

        # 結果をJSONファイルに保存
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file, ensure_ascii=False, indent=4)

    def save_processed_audio(self, output_path):
        sf.write(output_path, self.processed_audio, self.sr)

def main():
    # ジレクトリの名前は制定する
    directory_path = '/Users/qaz1214/Downloads/internlm2-kamen-rider-blade-roleplay/data/'

    # 使用示例
    audio_processor = DataProcessor(os.path.join(directory_path, 'raw/audio/bgm/Elements.mp3'))
    audio_processor.apply_noise_reduction(reduction_method='wiener', intensity=0.5)
    audio_processor.adjust_snr(20)  # Target an SNR of 20 dB
    audio_processor.apply_noise_reduction_if_needed(threshold_snr=10)  # 仅当SNR低于10dB时应用降噪
    audio_processor.save_processed_audio(os.path.join(directory_path, 'processed/wav/Elements.wav'))

    # 文字を確認する
    print('文字を確認しました')

    # ジレクトリを追加する
    output_txt_path = os.path.join(directory_path, 'processed/episodes_txt/')  # 出力ファイルのパスを設定

    # エピソード処理クラスの初期化
    proc = DataProcessor(output_txt_path)

    # すべてのファイルを処理
    proc.process_all_files(proc.file_path)

    # テキストファイルに出力
    output_txt_path = os.path.join(output_txt_path, 'combined_.txt')
    proc.export_to_txt(output_txt_path)
    # JSONファイルに出力
    output_json_path = os.path.join(output_txt_path, 'combined_.json')
    proc.prompt_engineer(output_txt_path, output_json_path)
    print("データのエクスポートが完了しました。")

if __name__ == "__main__":
    main()
