import sys
sys.path.append("../../")
from dlgo import agent
from server import get_web_app

myagent = agent.CNNBot()
web_app = get_web_app({'cnn': myagent})
web_app.run()
