# Third-party notices

InjecForks and the code in this repository reuse material from the projects below. Each is distributed under the MIT
licence; its notice is reproduced here.

| Project | What we use | Version | Licence |
|---|---|---|---|
| InjecAgent (Zhan et al., Findings of ACL 2024) | the 62 attacker goals, the 17 user tools and their tool definitions, the agent system prompt; InjecForks' cases are built from them | commit `f19c9f2c79a41046eb13c03c51a24c567a8ffa07` | MIT |
| AgentDojo (Debenedetti et al., NeurIPS 2024 Datasets and Benchmarks) | the injection templates `important_instructions_no_names` and `tool_knowledge` (InjecForks' two arms); the `agentdojo` package for the AgentDojo runs (installed from PyPI, not vendored) | `agentdojo==0.1.35`, commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60` | MIT |

Models and classifiers (Qwen3.5, DeepSeek-R1-Distill-Llama-8B, gpt-oss-20b, PromptGuard 2, PIGuard, e5-large-v2,
DeBERTa-v3-base) are downloaded at run time under their own licences and are not redistributed here. PromptGuard 2 is
under the Llama 4 Community licence and needs that licence accepted on Hugging Face.

## InjecAgent

```
MIT License

Copyright (c) 2023 Qiusi Zhan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## AgentDojo

```
MIT License

Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
