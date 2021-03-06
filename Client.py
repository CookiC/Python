import socket,time,threading

global end
global s
end=False
data=''
while(data==''):
	data=input('输入用户名：')
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 建立连接:
s.connect(('192.168.3.6', 2080))
s.settimeout(2)
s.send(data.encode('utf-8'))
# 接收欢迎消息:
print(s.recv(1024).decode('utf-8'))

def sendMessage():
	global end
	global s
	data=''
	while data!='end' and data!='exit':
		data=input()
		s.send(data.encode('utf-8'))
	end=True

def recvMessage():
	global end
	global s
	while not end:
		try:
			print(s.recv(1024).decode('utf-8'))
			if s.recv(1024).decode('utf-8')== 'end':
				break
		except socket.timeout:
			continue
	s.close()

sendt=threading.Thread(target=sendMessage)
recvt=threading.Thread(target=recvMessage)
sendt.start()
recvt.start()
while not end:
	time.sleep(1)