import requests
from bs4 import BeautifulSoup

def get_video_info(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 获取视频标题
        title = soup.find('h1').text.strip()
        
        # 获取视频描述
        description = soup.find('div', class_='video-description').text.strip()
        
        # 获取视频时长
        duration = soup.find('span', class_='video-duration').text.strip()
        
        # 获取视频播放链接
        video_url = soup.find('video')['src']
        
        return {
            'title': title,
            'description': description,
            'duration': duration,
            'video_url': video_url
        }
    except Exception as e:
        print(f"无法从 {url} 获取视频信息: {e}")

# 示例视频播放器链接
blade_url = "https://www.yhdmhy.com/_player_x_/592724"
# video_player_url = "https://example.com/video-player"

# 获取视频信息
video_info = get_video_info(video_player_url)
if video_info:
    print("视频标题:", video_info['title'])
    print("视频描述:", video_info['description'])
    print("视频时长:", video_info['duration'])
    print("视频播放链接:", video_info['video_url'])
