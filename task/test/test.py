# 使用shell模式运行命令
import subprocess as sp
result = sp.run(["ls -al; echo helloe"], shell=True, capture_output=False, text=True)
print(result)