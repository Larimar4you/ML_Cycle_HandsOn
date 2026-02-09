#!/bin/bash
# --- Пуш одним кликом через SSH ---

# 1️⃣ Запускаем ssh-agent
eval "$(ssh-agent -s)"

# 2️⃣ Добавляем ключ
ssh-add ~/.ssh/id_ed25519

# 3️⃣ Проверяем remote и ставим правильный SSH для текущего репо
git remote set-url origin git@github.com:Larimar4you/$(basename $(git rev-parse --show-toplevel)).git

# 4️⃣ Добавляем все изменения
git add .

# 5️⃣ Делаем коммит с авто-сообщением
git commit -m "Auto commit $(date +'%Y-%m-%d %H:%M:%S')"

# 6️⃣ Пушим в main
git push -u origin main

echo "🎉 Готово! Пуш выполнен через SSH"
