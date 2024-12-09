# llm-recommender
## ML System Design doc
Документ расположен по [ссылке](https://docs.google.com/document/d/1Ga691U9pWU0-KPTTrtEkKcKPsr2oaGiAG6fO8wdlmkI/edit?usp=sharing)

## PDM
Установить PDM можно по инструкции по ссылке https://pdm-project.org/en/latest/#installation  
Чтобы установить конкретную версию Python через PDM можно использовать следующую команду:
```bash
pdm python install 3.12  
```
Чтобы установить все зависимости, используйте:
```bash
pdm install
```
Эта команда установит все зависимости, зафиксированные в pyproject.toml
Чтобы добавить новую зависимость, используйте:
```bash
pdm add <package_name>
```

## Linters, formatters
Предлагается использовать Ruff для форматирования кода. В VS Code есть [плагин](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) для него. Автоформатирование для VS Code наcтроено в .vscode/settings.json. Это позволит автоматически форматировать файл каждый раз при сохранении.
В качестве линтеров используются pylint и flake8