from multiprocessing import Process
import socket,time,threading

global socks
global cmd
socks={}
cmd=""

def tcpLink(sock,addr):
	socks[addr]=sock
	name = sock.recv(1024).decode('utf-8')
	print("接收到来自%s:%s的连接..." % addr)
	sock.send(b'Welcome!')
	while True:
		data = sock.recv(1024)
		time.sleep(0.1)
		if not data or data.decode('utf-8') == 'exit':
			break
		if data.decode('utf-8') == 'end':
			cmd='end'
			break
		data=data.decode('utf-8')
		for e in socks:
			socks[e].send((name+' : '+data).encode('utf-8'))
	socks.pop(addr)

def commend():
	global cmd
	while cmd!='end':
		cmd=input()

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1',2080))
s.listen(5)
s.settimeout(5)
print("服务器酱开始工作啦！")
cmdt=threading.Thread(target=commend)
cmdt.start()
while cmd!='end':
	try:
		sock, addr = s.accept()
		t = threading.Thread(target=tcpLink, args=(sock, addr))
		t.start()
	except socket.timeout:
		print(time.strftime("%H:%M:%S", time.localtime())+'服务器酱等待中哟...')
#		print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())+'服务器酱等待中哟...')
s.close()
print("服务器酱睡觉去啦！")

