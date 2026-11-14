"""Blender（bpy）で 3D の文字を作り、左右に振れる動画（MP4）に書き出すサンプル。

使い方（tools/ ディレクトリで実行）:
  uv run python blender_render.py --text "動画編集" --out ../out/blender_sample.mp4
  uv run python blender_render.py --still --out ../out/blender_sample.png

ポイント:
  - 画面なし（ヘッドレス）で Blender を Python から直接動かす。Blender MCP は使わない。
  - GPU がないため Cycles の CPU 描画を使う。サンプル数と解像度で速さが決まる。
  - 日本語の文字はシステムの日本語フォント（IPAゴシック）を使う。
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import bpy


def find_japanese_font() -> str | None:
    """fontconfig で日本語に対応したフォントファイルを探す。"""
    if shutil.which("fc-match") is None:
        return None
    result = subprocess.run(
        ["fc-match", "-f", "%{file}", ":lang=ja"], capture_output=True, text=True, check=False
    )
    path = result.stdout.strip()
    return path if path and Path(path).exists() else None


def build_scene(text: str, color: tuple[float, float, float]) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bpy.ops.object.text_add(location=(0, 0, 0))
    obj = bpy.context.object
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = 1.25
    obj.data.extrude = 0.12
    obj.data.bevel_depth = 0.02
    font_path = find_japanese_font()
    if font_path:
        obj.data.font = bpy.data.fonts.load(font_path)
    obj.rotation_euler = (math.radians(90), 0, 0)

    material = bpy.data.materials.new("TextMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.4
    bsdf.inputs["Roughness"].default_value = 0.3
    obj.data.materials.append(material)

    obj.rotation_mode = "XYZ"

    bpy.ops.object.camera_add(location=(0, -7.5, 0.6), rotation=(math.radians(86), 0, 0))
    scene.camera = bpy.context.object

    bpy.ops.object.light_add(type="AREA", location=(3, -4, 4))
    bpy.context.object.data.energy = 900
    bpy.context.object.data.size = 4

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.025, 0.05, 1)
    scene.world = world


def main() -> None:
    parser = argparse.ArgumentParser(description="Blender で 3D 文字の動画を書き出す")
    parser.add_argument("--text", default="動画編集", help="3D にする文字")
    parser.add_argument("--out", type=Path, required=True, help="出力先（.mp4 または --still 時は .png）")
    parser.add_argument("--still", action="store_true", help="静止画 1 枚だけ書き出す")
    parser.add_argument("--frames", type=int, default=48, help="フレーム数（既定: 48）")
    parser.add_argument("--fps", type=int, default=24, help="フレームレート（既定: 24）")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=16, help="Cycles のサンプル数（多いほど綺麗で遅い）")
    args = parser.parse_args()

    build_scene(args.text, color=(1.0, 0.45, 0.2))
    scene = bpy.context.scene
    obj = next(o for o in scene.objects if o.type == "FONT")
    # 左右に振れるアニメーション（-25° → +25° → -25°）。文字が常に読める角度に収まる。
    scene.frame_start = 1
    scene.frame_end = args.frames
    middle = (scene.frame_start + scene.frame_end) // 2
    for frame, degrees in ((scene.frame_start, -25), (middle, 25), (scene.frame_end, -25)):
        obj.rotation_euler[2] = math.radians(degrees)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = args.fps

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.still:
        # 文字が正面を向くフレーム（振れ幅の中間）を書き出す。
        scene.frame_set(max(1, (scene.frame_start + scene.frame_end) // 4))
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(args.out.resolve())
        bpy.ops.render.render(write_still=True)
    else:
        # フレームを PNG で書き出し、システムの FFmpeg で MP4 にまとめる。
        frames_dir = args.out.with_suffix("").parent / (args.out.stem + "_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(frames_dir.resolve() / "frame_")
        bpy.ops.render.render(animation=True)
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps),
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(args.out),
            ],
            check=True,
        )
        shutil.rmtree(frames_dir)
    print(f"Blender {bpy.app.version_string} / 書き出し {time.perf_counter() - started:.1f} 秒 / 出力: {args.out}")


if __name__ == "__main__":
    main()
