import { Composition } from "remotion";
import { MyComposition } from "./Composition";
import { ThreeScene } from "./samples/ThreeScene";
import { TitleCard } from "./samples/TitleCard";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <MyComposition />
      <Composition
        id="SampleTitle"
        component={TitleCard}
        durationInFrames={90}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{
          title: "動画編集スタジオ",
          subtitle: "Remotion で日本語タイトルを動かすサンプル",
          accentColor: "#ff7a3d",
        }}
      />
      <Composition
        id="SampleThree"
        component={ThreeScene}
        durationInFrames={90}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{ color: "#ff7a3d" }}
      />
    </>
  );
};
