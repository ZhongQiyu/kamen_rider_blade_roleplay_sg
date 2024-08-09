# test_video.py

import unittest
from video_proc import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    def test_video_processing(self):
        video_path = "path_to_your_video.mp4"  # 替换为你的视频文件路径
        output_dir = "output_scenes"
        video_processor = VideoProcessor(video_path, output_dir)
        video_processor.process_video()

        # 检查输出文件是否存在
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'enhanced_video.mp4')))

if __name__ == "__main__":
    unittest.main()
