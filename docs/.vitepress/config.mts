import { defineConfig } from "vitepress";

export default defineConfig({
  title: "Praxis",
  description:
    "Documentation for comparing VLA methods, backbones, and action codecs",
  base: process.env.VITEPRESS_BASE ?? "/",
  cleanUrls: false,
  appearance: true,
  markdown: {
    math: true,
  },
  themeConfig: {
    nav: [
      { text: "Overview", link: "/" },
      { text: "Introduction", link: "/introduction" },
      { text: "Foundations", link: "/concepts/policy-io" },
      { text: "Codecs", link: "/codecs/overview" },
      { text: "Methods", link: "/methods/overview" },
      { text: "Quickstart", link: "/quickstart" },
      { text: "GitHub", link: "https://github.com/Chaoqi-LIU/praxis" },
    ],
    sidebar: [
      {
        text: "Overview",
        items: [
          { text: "Praxis", link: "/" },
          { text: "Introduction", link: "/introduction" },
          { text: "Installation", link: "/installation" },
          { text: "Quickstart", link: "/quickstart" },
        ],
      },
      {
        text: "Foundations",
        items: [
          { text: "PolicyIO Contract", link: "/concepts/policy-io" },
          { text: "Comparison Axes", link: "/concepts/comparison-axes" },
          { text: "Policy Matrix", link: "/concepts/policy-matrix" },
          { text: "System Boundary", link: "/concepts/eval-boundary" },
          { text: "Source Layout", link: "/concepts/repo-map" },
          { text: "Configuration", link: "/concepts/config-wiring" },
        ],
      },
      {
        text: "Action Representations",
        items: [
          { text: "Codec Interface", link: "/codecs/overview" },
          { text: "Binning", link: "/codecs/binning" },
          { text: "FAST", link: "/codecs/fast" },
          { text: "OAT", link: "/codecs/oat" },
          { text: "BOAT", link: "/codecs/boat" },
          { text: "QueST", link: "/codecs/quest" },
          { text: "ActionCodec", link: "/codecs/acodec" },
          { text: "Codec Training", link: "/codecs/training" },
        ],
      },
      {
        text: "Policy Objectives",
        items: [
          { text: "Overview", link: "/methods/overview" },
          { text: "BAR", link: "/methods/bar" },
          { text: "FM", link: "/methods/fm" },
          { text: "KI", link: "/methods/ki" },
        ],
      },
      {
        text: "Backbone Adapters",
        items: [
          { text: "Overview", link: "/backbones/overview" },
          { text: "PaliGemma2", link: "/backbones/paligemma" },
          { text: "Qwen3-VL", link: "/backbones/qwen3" },
          { text: "SmolVLM", link: "/backbones/smolvlm" },
        ],
      },
      {
        text: "Experimental Workflows",
        items: [
          { text: "BAR OAT Pali Example", link: "/guides/bar-oat-pali-example" },
          { text: "Train A Policy", link: "/guides/train-policy" },
          { text: "Evaluate A Policy", link: "/guides/evaluate-policy" },
          { text: "Serve A Policy", link: "/guides/serve-policy" },
        ],
      },
      {
        text: "Extension",
        items: [
          { text: "New Action Codec", link: "/development/adding-codecs" },
          { text: "New VLM Backbone", link: "/development/adding-backbones" },
          { text: "New Policy Method", link: "/development/adding-policy-variants" },
        ],
      },
      {
        text: "Reference",
        items: [
          { text: "CLI", link: "/reference/cli" },
          { text: "Policy Factory", link: "/reference/policy-factory" },
          { text: "Codec Factory", link: "/reference/codec-factory" },
          { text: "Training Artifacts", link: "/reference/training-artifacts" },
          { text: "Documentation Checks", link: "/development/docs-checks" },
          { text: "Publishing Docs", link: "/development/publishing" },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/Chaoqi-LIU/praxis" },
    ],
    search: {
      provider: "local",
    },
    footer: {
      message: "Released under the Apache-2.0 License.",
      copyright: "Copyright Chaoqi Liu and contributors.",
    },
  },
});
