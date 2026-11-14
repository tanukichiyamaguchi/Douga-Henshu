import { ThreeCanvas } from "@remotion/three";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";

type Props = {
  readonly color: string;
};

// 動作確認用の 3D サンプル。回転は useCurrentFrame() だけで決める
// （@react-three/fiber の useFrame() は書き出し時にちらつくため使わない）。
export const ThreeScene: React.FC<Props> = ({ color }) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();

  return (
    <AbsoluteFill style={{ backgroundColor: "#0b1020" }}>
      <ThreeCanvas
        width={width}
        height={height}
        camera={{ fov: 50, position: [0, 0, 4.2] }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[3, 3, 3]} intensity={2} />
        <mesh rotation={[frame * 0.03, frame * 0.06, 0]}>
          <torusKnotGeometry args={[1, 0.35, 200, 32]} />
          <meshStandardMaterial color={color} metalness={0.4} roughness={0.3} />
        </mesh>
      </ThreeCanvas>
    </AbsoluteFill>
  );
};
