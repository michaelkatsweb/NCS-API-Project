# Emergency Code Formatting Fix
Write-Host "🚨 EMERGENCY: Fixing code formatting to stop CI/CD failures..." -ForegroundColor Red

# Install formatting tools
Write-Host "Installing formatting tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install black==23.11.0 isort==5.12.0 flake8==6.1.0

# Auto-fix all formatting issues
Write-Host "Applying Black formatter..." -ForegroundColor Yellow
python -m black .

Write-Host "Sorting imports..." -ForegroundColor Yellow
python -m isort .

# Check what changed
Write-Host "Checking changes..." -ForegroundColor Yellow
git status

Write-Host "✅ Code formatting applied! Ready to commit." -ForegroundColor Green
Write-Host ""
Write-Host "Next: Run these commands to commit:" -ForegroundColor Cyan
Write-Host "git add ." -ForegroundColor White
Write-Host "git commit -m 'fix: auto-format code to resolve CI/CD failures'" -ForegroundColor White
Write-Host "git push" -ForegroundColor White