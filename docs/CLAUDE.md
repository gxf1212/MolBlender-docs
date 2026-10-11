

### Changelog管理
- **CHANGELOG.md维护**: docs/source的CHANGELOG.md是唯一官方变更记录来源
- **格式标准**: 遵循[Keep a Changelog](https://keepachangelog.com/)格式，使用语义化版本
- **实时更新**: 每次功能开发或bug修复都必须同步更新CHANGELOG.md的[Unreleased]部分. write into CHANGELOG every time we git push.
- make html for the new doc every time the docs change. don't forget to push the docs repo too.
- **分类记录**: 变更分为Added(新功能)、Changed(功能修改)、Deprecated(即将弃)、Removed(已删除)、Fixed(bug修复)、Security(安全修复)
- **提交规范**: 使用Conventional Commits格式，如`feat(dashboard): add new chart`、`fix(models): resolve compatibility issue`
- **版本发布**: 发布新版本时，将[Unreleased]内容移至带日期的版本标签下，如`## [0.2.0] - 2025-10-22`


### build and write

每次改完文档都make html一下，记得新page加到sidebar toc里面

To build the docs, run: `make html`

- try to write .md if possible for usage and rst for API (if no complex notes). 
- one canvas per file 
- follows sphinx and myst as markdown parser
- you can use sphinx-design to beautify (but not full of them), e.g. use Grids, Cards, Dropdowns, Tabs, Badges, Buttons & Icons. cards are for display of various submodules. tabs might be used for switching modes
- and also myst Admonitions
- no try except in ``` code blocks since user will test themselves
- similarly, no if emb is not None:print(f"CLS Rep Shape for '{smiles_batch[i]}': {emb.shape}"). just directly print...
- try to list all featurizers (parameters) in this submodule in a table. and users can easily copy and paste keys into the example code to use
- Minimize empty lines to improve readability without sacrificing clarity.
- abundant comments, like possible output format
- clear and simple for users to understand. the first principle is user-friendly. users read this and quickly write their own application
- use $ around equations
- keep the title concise since they appear in the doctree! just the keyword.
- put dependency installation after introduction (if any; no need for required dependencies like rdkit). deepchem/mordred/cdk/molfeat are optional extras; a single bash code block is enough
- no unrelated usages, just all possible fundamental usage about this file
- add links to original software packages (related doc/repo) for reference in the end
- do not use markers like {py:func} in inside a table
- but always use sth like this to refer to api: {doc}`/api/representations/fingerprints/deepchem`
- always add this to accept toctree (order of subpages are optional)
```{toctree}
:maxdepth: 1
:hidden:
```

for submodule index pages, should use this to list individual pages
```{toctree}
:maxdepth: 1
:hidden:

rdkit
cdk
deepchem
datamol
```

### Avoiding TOC Duplication

**Two Rules**:
1. **In `.md` files**: Always use `:maxdepth: 1` and `:hidden:` in toctrees
2. **In `index.rst`**: Only list parent pages, NOT their children (children already in parent's toctree)

**Example** - `index.rst` should only reference parent:
```rst
.. toctree::
   :caption: User Guide

   usage/index          ← ✅ Parent only
```

NOT:
```rst
.. toctree::
   :caption: User Guide

   usage/index
   usage/representations/index  ← ❌ Child (already in usage/index toctree)
   usage/data/index             ← ❌ Child
```

**Why**:
- `:maxdepth: 2+` renders nested TOCs as page content
- Listing children in `index.rst` duplicates sidebar navigation

**Checklist**:
- ✅ All `.md` toctrees: `:maxdepth: 1` + `:hidden:`
- ✅ `index.rst`: only parent pages
- ✅ After changes: `make clean && make html`
- ✅ Hard refresh browser (Ctrl+Shift+R) to clear cache
