import { loadFont } from "@remotion/google-fonts/NotoSansJP";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

// Google Fonts の日本語フォントは、太さ 1 種類あたり約 120 個のファイルに分割されている。
// 使う太さだけを読み込み、「通信回数が多い」警告は日本語では避けられないため止める。
const { fontFamily } = loadFont("normal", {
  weights: ["400", "700"],
  subsets: ["japanese", "latin"],
  ignoreTooManyRequestsWarning: true,
});

type Props = {
  readonly title: string;
  readonly subtitle: string;
  readonly accentColor: string;
};

// 動作確認用のサンプル。アニメーションはすべて useCurrentFrame() から計算する。
export const TitleCard: React.FC<Props> = ({ title, subtitle, accentColor }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const rise = spring({ frame, fps, config: { damping: 200 } });
  const titleY = interpolate(rise, [0, 1], [40, 0]);
  const subtitleOpacity = interpolate(frame, [15, 35], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const barWidth = interpolate(frame, [5, 30], [0, 320], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#0e1116",
        justifyContent: "center",
        alignItems: "center",
        fontFamily,
      }}
    >
      <div style={{ textAlign: "center" }}>
        <h1
          style={{
            color: "#ffffff",
            fontSize: 110,
            fontWeight: 700,
            margin: 0,
            opacity: rise,
            transform: `translateY(${titleY}px)`,
          }}
        >
          {title}
        </h1>
        <div
          style={{
            height: 8,
            width: barWidth,
            backgroundColor: accentColor,
            margin: "32px auto",
            borderRadius: 4,
          }}
        />
        <p
          style={{
            color: "#c9d1d9",
            fontSize: 44,
            fontWeight: 400,
            margin: 0,
            opacity: subtitleOpacity,
          }}
        >
          {subtitle}
        </p>
      </div>
    </AbsoluteFill>
  );
};
