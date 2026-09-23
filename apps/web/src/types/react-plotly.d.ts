declare module "react-plotly.js" {
  import type { CSSProperties, Component } from "react";
  import type { Config, Data, Layout } from "plotly.js";

  interface PlotParams {
    data: Data[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    style?: CSSProperties;
    className?: string;
  }

  export default class Plot extends Component<PlotParams> {}
}
