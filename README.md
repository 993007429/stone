# stone

项目代号：维克·斯通（Stone）

本仓库是数据、算法管理系统（STONE）的后端代码仓库 

* Python3.8 + Flask2.3.2 + MySQL5.7

开发时可以直接使用 `python run.py` 运行本地 Server。

## 依赖

使用 [pip-tools](https://github.com/jazzband/pip-tools/) 管理依赖。

安装静态依赖（版本变化需要告知运维升级
```
pip-compile
pip install requirements.txt
```

添加、删除、升级依赖时，首先修改原始的 `requirements.in` 文件（间接依赖不必写入这个文件），然后运行 `pip-compile` 更新 `requirements.txt`，最后通过 `pip-sync` 或者 `pip install -r requirements.txt` 更新虚拟环境。

# 代码规范
- 代码风格参考 [PEP8](https://www.python.org/dev/peps/pep-0008/) ,部分如下:
  - 缩进使用 4个 `space`；（尽量使用单引号吧，目前整个项目都是单引号:-）
  - 包引用
    - 优先标准库
    - 其次相关第三方包
    - 最后是应用的内部包
    - 包引用顺序按字母升序
    - 尽量避免 _wildcard imports_，即 `import *`
  - [`doc string`](https://www.python.org/dev/peps/pep-0008/#documentation-strings) 使用三引号
  - 尽量添加[函数注解](https://www.python.org/dev/peps/pep-0008/#function-annotations)
  > 项目根目录执行: `flake8 && pylint seal` 进行校验
- 版本发布遵循 [Semantic Versioning](https://semver.org/#semantic-versioning-specification-semver)

# shell 调试
- 可通过根目录执行： python shell.py 可进入项目交互环境
- 提供项目内的包环境，自动补全，提示。
- 例如: `from stone.infra import session`

## Q&A

待写
