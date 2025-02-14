import time
import asyncio
from pkg.plugin.context import register, handler, BasePlugin, APIHost, EventContext
from pkg.plugin.events import *  # 导入事件类
import os
import requests
import httpx
import logging
import re
from queue import Queue, Empty
from mirai import Image, MessageChain,Plain,Voice,Plain

import json
import wave
from pydub import AudioSegment
import pyaudio
from audio_separator.separator import Separator
import subprocess
from graiax import silkcoder

token = 'YOURTOKEN'  # 请将这里的'YOUR_TOKEN'替换为你实际获取的token
cookie = "YOUR_COOKIE"  # 请将这里的'YOUR_COOKIE'替换为你实际获取的cookie
RVC_logs_path = r"F:\RVC\RVC1006Nvidia\logs"  # 请将这里的"F:\RVC\RVC1006Nvidia\logs"替换为你部署的RVC项目的logs文件夹的路径

new_mdx_params = {"hop_length": 1024, "segment_size": 256, "overlap": 8, "batch_size": 4, "enable_denoise": False}
new_vr_params = {"batch_size": 4, "window_size": 512, "aggression": 5, "enable_tta": False, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False}


# 注册插件
@register(name="RVC_Music", description="RVC翻唱音乐", version="0.1", author="zzseki")
class RVC_Music(BasePlugin):
    # 插件加载时触发
    def __init__(self, host: APIHost):
        self.logger = logging.getLogger(__name__)

    # 当收到群消息时触发
    @handler(GroupNormalMessageReceived)
    async def group_Normal_message_received(self, ctx: EventContext):
        receive_text = ctx.event.text_message
        if "#学习翻唱" in receive_text:
            ctx.prevent_default()
            pattern = re.compile(r"\[(.*?)\]\[([-+]?\d+)\]\[(.+?)\]")
            match = pattern.search(receive_text)
            if match:
                music = match.group(1)  # 提取第一个方括号内的内容
                f0up = int(match.group(2))  # 提取第二个方括号内的内容
                model_name = match.group(3)  # 提取第三个方括号内的内容
                dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
                wav_path = os.path.join(dir_path, f'{music}.wav')
                # 确保目录存在
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                id, artists, music_name = self.get_music_id(music)
                if id:
                    msg, url = self.get_music(id)
                    if msg != "success":
                        self.ap.logger.info(f"{music_name} {artists}", msg)
                        id, artists, music_name = self.get_music_id(music, 1)
                        if id:
                            msg, url = self.get_music(id)
                    if url:
                        music_name = music_name.replace('/', '&')
                        music_name = music_name.replace('"', '_')
                        music_name = music_name.replace("'", ' ')
                        music_name = music_name.replace(":", ' ')
                        music_name = music_name.replace("：", ' ')
                        artists = artists.replace('/', '&')
                        wav_path = self.download_audio(url, music_name, artists)
                        await ctx.event.query.adapter.reply_message(ctx.event.query.message_event, [(f"正在学习 {music_name} {artists}......")], False)
                        try:
                            banzou_path, hesheng_path = self.UVR5(f"{music_name} {artists}")
                            if not self.is_pcm_s16le(banzou_path):
                                print(f"伴奏 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(banzou_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(banzou_path, output_path)
                            if not self.is_pcm_s16le(hesheng_path):
                                print(f"和声 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(hesheng_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(hesheng_path, output_path)
                            dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
                            music_path = os.path.join(dir_path, f"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR_(No Reverb)_UVR-DeEcho-DeReverb.wav")
                            gansheng_path = self.send_request(model_name, music_path, f"{music_name} {artists}", int(f0up))
                            if not self.is_pcm_s16le(gansheng_path):
                                print(f"干声 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(gansheng_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(gansheng_path, output_path)
                            # 加载干声
                            dry_vocals = AudioSegment.from_wav(gansheng_path)
                            dry_vocals = dry_vocals + 3

                            # 保存临时干声音频
                            dry_vocals.export(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_temp.wav'), format="wav")

                            # 使用 ffmpeg 为干声增加混响效果
                            subprocess.call([
                                'ffmpeg', '-i', os.path.join(dir_path, f'{music_name} {artists} dry_vocals_temp.wav'),
                                '-filter_complex', 'aecho=0.5:0.7:60:0.2',  # 混响参数
                                os.path.join(dir_path, f'{music_name} {artists} dry_vocals_with_reverb.wav')
                            ])
                            # 衰减量：第一个参数（0.8）控制输入信号的衰减，通常设置得低一点会减少混响强度。
                            # 回声衰减：第二个参数（0.88）控制回声信号的衰减，增大这个值可以减少混响的明显程度。
                            # 延迟时间：第三个参数（60）是回声的延迟时间，可以根据需要保持不变或进行微调。
                            # 反馈量：第四个参数（0.4）控制反馈强度，降低这个值会减弱混响效果。

                            # 加载处理后的干声
                            reverb_vocals = AudioSegment.from_wav(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_with_reverb.wav'))

                            # 加载伴奏和和声
                            accompaniment = AudioSegment.from_wav(banzou_path)
                            harmony = AudioSegment.from_wav(hesheng_path)

                            # 合并音轨
                            combined = accompaniment.overlay(harmony).overlay(reverb_vocals)
                            RVC_Music_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RVC_Music")
                            output_file_path = os.path.join(RVC_Music_path, f"{music_name} {artists}_{model_name}.wav")
                            # 导出最终合成的歌曲
                            combined.export(output_file_path, format="wav")
                            if not self.is_pcm_s16le(output_file_path):
                                print(f"伴奏 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(output_file_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(output_file_path, output_path)
                            silk_path = self.convert_to_silk(model_name, output_file_path, f"{music_name} {artists}")
                            ctx.add_return("reply", [Voice(path=str(silk_path))])
                            # 删除临时文件
                            os.remove(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_temp.wav'))
                            os.remove(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_with_reverb.wav'))
                            os.remove(gansheng_path)
                            os.remove(hesheng_path)
                            os.remove(banzou_path)
                            os.remove(os.path.join(dir_path,
                                                   fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12.wav"))
                            os.remove(os.path.join(dir_path,
                                                   fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR.wav"))
                            os.remove(os.path.join(dir_path,
                                                   fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR_(No Reverb)_UVR-DeEcho-DeReverb.wav"))
                            os.remove(os.path.join(dir_path,
                                                   fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR_(Reverb)_UVR-DeEcho-DeReverb.wav"))
                            #os.remove(silk_path)
                            os.remove(wav_path)
                            os.remove(output_file_path)
                            ctx.prevent_default()
                        except:
                            await ctx.event.query.adapter.reply_message(ctx.event.query.message_event, [(f"出错啦！ •᷄ࡇ•᷅")], False)
                            ctx.prevent_default()
                else:
                    self.ap.logger.info("提取音乐名称失败")

    @handler(PersonNormalMessageReceived)
    async def person_normal_message_received(self, ctx: EventContext, **kwargs):
        receive_text = ctx.event.text_message
        if "#学习翻唱" in receive_text:
            pattern = re.compile(r"\[(.*?)\]\[([-+]?\d+)\]\[(.+?)\]")
            match = pattern.search(receive_text)
            if match:
                music = match.group(1)  # 提取第一个方括号内的内容
                f0up = int(match.group(2))  # 提取第二个方括号内的内容
                model_name = match.group(3)  # 提取第三个方括号内的内容
                dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
                wav_path = os.path.join(dir_path, f'{music}.wav')
                # 确保目录存在
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                id, artists, music_name = self.get_music_id(music)
                if id:
                    msg, url = self.get_music(id)
                    if msg != "success":
                        self.ap.logger.info(f"{music_name} {artists}", msg)
                        id, artists, music_name = self.get_music_id(music, 1)
                        if id:
                            msg, url = self.get_music(id)
                    if url:
                        music_name = music_name.replace('/', '&')
                        music_name = music_name.replace('"', '_')
                        music_name = music_name.replace("'", ' ')
                        music_name = music_name.replace(":", ' ')
                        music_name = music_name.replace("：", ' ')
                        artists = artists.replace('/', '&')
                        wav_path = self.download_audio(url, music_name, artists)
                        await ctx.event.query.adapter.reply_message(ctx.event.query.message_event, [(f"正在学习 {music_name} {artists}......")], False)
                        try:
                            banzou_path, hesheng_path = self.UVR5(f"{music_name} {artists}")
                            if not self.is_pcm_s16le(banzou_path):
                                print(f"伴奏 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(banzou_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(banzou_path, output_path)
                            if not self.is_pcm_s16le(hesheng_path):
                                print(f"和声 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(hesheng_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(hesheng_path, output_path)
                            dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
                            music_path = os.path.join(dir_path, f"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR_(No Reverb)_UVR-DeEcho-DeReverb.wav")
                            gansheng_path = self.send_request(model_name, music_path, f"{music_name} {artists}", int(f0up))
                            if not self.is_pcm_s16le(gansheng_path):
                                print(f"干声 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(gansheng_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(gansheng_path, output_path)
                            # 加载干声
                            dry_vocals = AudioSegment.from_wav(gansheng_path)
                            dry_vocals = dry_vocals + 3

                            # 保存临时干声音频
                            dry_vocals.export(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_temp.wav'), format="wav")

                            # 使用 ffmpeg 为干声增加混响效果
                            subprocess.call([
                                'ffmpeg', '-i', os.path.join(dir_path, f'{music_name} {artists} dry_vocals_temp.wav'),
                                '-filter_complex', 'aecho=0.5:0.7:60:0.2',  # 混响参数
                                os.path.join(dir_path, f'{music_name} {artists} dry_vocals_with_reverb.wav')
                            ])
                            # 衰减量：第一个参数（0.8）控制输入信号的衰减，通常设置得低一点会减少混响强度。
                            # 回声衰减：第二个参数（0.88）控制回声信号的衰减，增大这个值可以减少混响的明显程度。
                            # 延迟时间：第三个参数（60）是回声的延迟时间，可以根据需要保持不变或进行微调。
                            # 反馈量：第四个参数（0.4）控制反馈强度，降低这个值会减弱混响效果。

                            # 加载处理后的干声
                            reverb_vocals = AudioSegment.from_wav(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_with_reverb.wav'))

                            # 加载伴奏和和声
                            accompaniment = AudioSegment.from_wav(banzou_path)
                            harmony = AudioSegment.from_wav(hesheng_path)

                            # 合并音轨
                            combined = accompaniment.overlay(harmony).overlay(reverb_vocals)
                            RVC_Music_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RVC_Music")
                            output_file_path = os.path.join(RVC_Music_path, f"{music_name} {artists}_{model_name}.wav")
                            # 导出最终合成的歌曲
                            combined.export(output_file_path, format="wav")
                            if not self.is_pcm_s16le(output_file_path):
                                print(f"伴奏 不是 16 位 PCM 格式，正在转换...")
                                output_path = os.path.splitext(output_file_path)[0] + "_16bit.wav"
                                self.convert_to_pcm_s16le(output_file_path, output_path)
                            silk_path = self.convert_to_silk(model_name, output_file_path, f"{music_name} {artists}")
                            ctx.add_return("reply", [Voice(path=str(silk_path))])
                            # 删除临时文件
                            os.remove(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_temp.wav'))
                            os.remove(os.path.join(dir_path, f'{music_name} {artists} dry_vocals_with_reverb.wav'))
                            os.remove(gansheng_path)
                            os.remove(hesheng_path)
                            os.remove(banzou_path)
                            os.remove(os.path.join(dir_path, fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12.wav"))
                            os.remove(os.path.join(dir_path, fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR.wav"))
                            os.remove(os.path.join(dir_path, fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR_(No Reverb)_UVR-DeEcho-DeReverb.wav"))
                            os.remove(os.path.join(dir_path, fr"{music_name} {artists}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR_(Reverb)_UVR-DeEcho-DeReverb.wav"))
                            #os.remove(silk_path)
                            os.remove(wav_path)
                            os.remove(output_file_path)
                            ctx.prevent_default()
                        except:
                            await ctx.event.query.adapter.reply_message(ctx.event.query.message_event, [(f"出错啦！ •᷄ࡇ•᷅")], False)
                            ctx.prevent_default()
                else:
                    self.ap.logger.info("提取音乐名称失败")


    def get_music_id(self, music_name, i=0):
        url = "https://v2.alapi.cn/api/music/search"
        params = {
            "keyword": music_name,
            "token": token,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()["data"]
            song_id = data['songs'][i]['id']
            artists = data['songs'][i]['artists'][0]['name']
            get_music_name = data['songs'][i]['name']
            return song_id, artists, get_music_name
        except httpx.HTTPStatusError as e:
            self.ap.logger.info(f"获取音乐 id 失败:" + str(e))
            return None

    def get_music(self, id):
        time.sleep(2)
        url = "https://v2.alapi.cn/api/music/url"
        params = {
            "id": id,
            "format": "json",
            "token": token,
            'cookie': cookie,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()["data"]
        msg = response.json()["message"]
        if data:
            url = data["url"]
            return msg, url
        else:
            url = None
            return msg, url

    def download_audio(self, audio_url, music_name, artists):
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
        mp3_path = os.path.join(dir_path, f"{music_name} {artists}.mp3")
        wav_path = os.path.join(dir_path, f"{music_name} {artists}.wav")
        flac_path = os.path.join(dir_path, f"{music_name} {artists}.flac")
        if re.search("flac", audio_url):
            file_type = "flac"
            file_path = flac_path
        elif re.search("mp3", audio_url):
            file_type = "mp3"
            file_path = mp3_path
        else:
            file_type = "wav"
            file_path = wav_path
        try:
            response = requests.get(audio_url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                self.ap.logger.info(f"音频文件已成功保存为" + file_path)
                try:
                    # 加载 flac 文件
                    audio = AudioSegment.from_file(file_path, format=file_type)
                    # 导出为 WAV 格式
                    audio.export(wav_path, format="wav")
                    self.ap.logger.info(f"文件已成功从 {file_type} 转换为 WAV 并保存为 {wav_path}")
                    # 删除 FLAC 文件
                    os.remove(file_path)
                    return wav_path
                except Exception as e:
                    self.ap.logger.info(f"转换音频文件发生异常: {str(e)}")
                    return False
            else:
                self.ap.logger.info(f"下载音频文件失败，状态码{response.status_code}")
                return False
        except Exception as e:
            self.ap.logger.info(f"下载音频文件发生异常" + str(e))
            return False

    def convert_to_silk(self,model_name, wav_path: str, name: str) -> str:
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
        silk_path = os.path.join(dir_path, f"{name} {model_name}.silk")
        if os.path.exists(silk_path):
            os.remove(silk_path)
            time.sleep(0.1)
        silkcoder.encode(wav_path, silk_path)

        # print(f"已将 WAV 文件 {wav_path} 转换为 SILK 文件 {silk_path}")
        return silk_path

    def UVR5(self, music_name):
        # Initialize the Separator class (with optional configuration properties, below)
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
        tmp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        models_path = os.path.join(tmp_path, "audio-separator-models")
        separator = Separator(output_dir=dir_path, model_file_dir=models_path, mdx_params=new_mdx_params, vr_params=new_vr_params)
        music_name = music_name
        # 1、去伴奏
        # separator.output_single_stem = "Vocals"
        separator.load_model("model_bs_roformer_ep_368_sdr_12.9628.ckpt")
        # Perform the separation on specific audio files without reloading the model
        output_files = separator.separate(os.path.join(dir_path, f"{music_name}.wav"))
        print(f"Separation complete! Output file(s): {' '.join(output_files)}")

        # 2、去和声
        # 6_HP-Karaoke-UVR.pth 少激进
        # 5_HP-Karaoke-UVR.pth 多激进
        separator.load_model("5_HP-Karaoke-UVR.pth")
        # Perform the separation on specific audio files without reloading the model
        output_files = separator.separate(os.path.join(dir_path, f"{music_name}_(Vocals)_model_bs_roformer_ep_368_sdr_12.wav"))
        print(f"Separation complete! Output file(s): {' '.join(output_files)}")

        # 3、去混响
        # UVR-De-Echo-Normal.pth 少量混响
        # UVR-De-Echo-Aggressive.pth 中等混响
        # UVR-DeEcho-DeReverb.pth  大量混响/正常混响
        # separator.output_single_stem = "Vocals"
        separator.load_model("UVR-DeEcho-DeReverb.pth")
        # Perform the separation on specific audio files without reloading the model
        output_files = separator.separate(os.path.join(dir_path, f"{music_name}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR.wav"))
        print(f"Separation complete! Output file(s): {' '.join(output_files)}")

        # # 4、降噪
        # # UVR-DeNoise.pth 降噪
        # separator.output_dir = r"F:\music\5jiangzao"
        # separator.output_single_stem = "Vocals"
        # # Load a machine learning model (if unspecified, defaults to 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt')
        # separator.load_model("UVR-DeNoise.pth")
        # # Perform the separation on specific audio files without reloading the model
        # output_files = separator.separate(fr"F:\music\4无混响干声_混响\{music_name}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Vocals)_5_HP-Karaoke-UVR_(No Reverb)_UVR-DeEcho-DeReverb.wav")
        # print(f"Separation complete! Output file(s): {' '.join(output_files)}")
        banzou_path = os.path.join(dir_path, f"{music_name}_(Instrumental)_model_bs_roformer_ep_368_sdr_12.wav")
        hesheng_path = os.path.join(dir_path, f"{music_name}_(Vocals)_model_bs_roformer_ep_368_sdr_12_(Instrumental)_5_HP-Karaoke-UVR.wav")
        return banzou_path, hesheng_path

    def send_request(self,model_name, file_path, music_name, f0up):
        index_path = self.get_index(model_name)
        url = "http://localhost:8001/voice2voice"
        params = {
            "model_name": f"{model_name}.pth",
            "index_path": index_path,
            "f0up_key": f0up,
            "f0method": "rmvpe",
            "index_rate": 0.66,
            "device": "cuda",
            "is_half": "False",
            "filter_radius": 3,
            "resample_sr": 0,
            "rms_mix_rate": 0.25,
            "protect": 0.33
        }

        # 使用 requests.Session 发送请求
        with requests.Session() as session:
            with open(file_path, "rb") as f:
                files = {"input_file": f}
                try:
                    response = session.post(url, files=files, params=params, stream=True)
                    response.raise_for_status()  # 检查请求是否成功
                    print(f"Status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"请求失败: {e}")
                    return None

        # 保存响应内容为 .wav 文件
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "music")
        output_file_path = os.path.join(dir_path, f"临时干声{music_name}_{model_name}.wav")
        try:
            with open(output_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 过滤保持活跃的块
                        f.write(chunk)
            print(f"文件已保存到: {output_file_path}")
        except IOError as e:
            print(f"保存文件失败: {e}")
            return None
        return output_file_path

    def get_index(self, model_name):
        # 目标文件夹路径
        folder_path = os.path.join(RVC_logs_path, model_name)

        # 获取文件夹中的所有文件
        all_items = os.listdir(folder_path)

        # 过滤只获取文件（排除子文件夹），并获取文件的完整路径
        file_paths = [os.path.join(folder_path, f) for f in all_items if os.path.isfile(os.path.join(folder_path, f))]

        # 假设只有一个文件，获取该文件的路径
        if file_paths:
            file_path = file_paths[0]
            return file_path
        else:
            print("获取index文件失败")
            return None

    def is_pcm_s16le(self, file_path):
        """检查 .wav 文件是否为 16 位 PCM (pcm_s16le)"""
        try:
            with wave.open(file_path, 'rb') as wf:
                sample_width = wf.getsampwidth()
                return sample_width == 2  # 16位音频每个采样点占用2字节
        except wave.Error:
            return False

    def convert_to_pcm_s16le(self, input_path, output_path):
        """使用 ffmpeg 将 .wav 文件转换为 16 位 PCM 格式"""
        try:
            # 使用 ffmpeg 进行转换
            subprocess.run([
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le',  # 转换为 16 位 PCM
                output_path
            ], check=True)
            # 删除原文件
            os.remove(input_path)

            # 将转换后的文件重命名为原文件名
            os.rename(output_path, input_path)
            print(f"文件转换并替换成功: {input_path}")
        except subprocess.CalledProcessError as e:
            print(f"转换失败: {e}")

    # 插件卸载时触发
    def __del__(self):
        pass
