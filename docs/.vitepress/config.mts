import { defineConfig } from "vitepress";

export default defineConfig({
  title: "Praxis",
  description: "Technical documentation for controlled VLA policy research",
  base: process.env.VITEPRESS_BASE ?? "/",
  cleanUrls: false,
  appearance: true,
  markdown: {
    math: true,
  },
  themeConfig: {
    nav: [
      { text: "Start", link: "/introduction" },
      { text: "Foundations", link: "/foundations/overview" },
      { text: "Axes", link: "/axes/overview" },
      { text: "Workflows", link: "/workflows/overview" },
      { text: "Extension", link: "/extension/overview" },
      { text: "Reference", link: "/reference/cli" },
      { text: "GitHub", link: "https://github.com/Chaoqi-LIU/praxis-vla" },
    ],
    sidebar: [
      {
        text: "Start Here",
        items: [
          { text: "Introduction", link: "/introduction" },
          { text: "Installation", link: "/installation" },
          { text: "Quickstart", link: "/quickstart" },
        ],
      },
      {
        text: "Foundations",
        items: [
          { text: "Overview", link: "/foundations/overview" },
          { text: "Policy Cells", link: "/foundations/policy-cells" },
          { text: "PolicyIO", link: "/foundations/policy-io" },
          { text: "Evaluation Boundary", link: "/foundations/evaluation-boundary" },
          { text: "Source Layout", link: "/foundations/source-layout" },
        ],
      },
      {
        text: "Design Axes",
        items: [
          { text: "Overview", link: "/axes/overview" },
          { text: "Objectives", link: "/axes/objectives" },
          { text: "Action Representations", link: "/axes/action-representations" },
          { text: "Backbones", link: "/axes/backbones" },
          { text: "Comparison Design", link: "/axes/comparison-design" },
        ],
      },
      {
        text: "Workflows",
        items: [
          { text: "Overview", link: "/workflows/overview" },
          { text: "Train A Codec", link: "/workflows/train-codec" },
          { text: "Train A Policy", link: "/workflows/train-policy" },
          { text: "Evaluate", link: "/workflows/evaluate" },
          { text: "Serve", link: "/workflows/serve" },
          { text: "Slurm Launchers", link: "/workflows/slurm-launchers" },
          { text: "BAR OAT Pali Path", link: "/workflows/bar-oat-pali" },
          { text: "G1 / Quest 3S Handoff", link: "/workflows/unitree_g1_handoff" },
        ],
      },
      {
        text: "Extension",
        items: [
          { text: "Overview", link: "/extension/overview" },
          { text: "Add A Codec", link: "/extension/add-codec" },
          { text: "Add A Backbone", link: "/extension/add-backbone" },
          { text: "Add An Objective", link: "/extension/add-objective" },
          { text: "Add A Public Cell", link: "/extension/add-public-cell" },
          { text: "Add A Cluster Launcher", link: "/extension/add-cluster-launcher" },
        ],
      },
      {
        text: "Reference",
        items: [
          { text: "CLI", link: "/reference/cli" },
          { text: "Configuration", link: "/reference/configuration" },
          { text: "Artifacts", link: "/reference/artifacts" },
          { text: "Policy Factory", link: "/reference/policy-factory" },
          { text: "Codec Factory", link: "/reference/codec-factory" },
          { text: "Docs Maintenance", link: "/reference/docs-maintenance" },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/Chaoqi-LIU/praxis-vla" },
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
