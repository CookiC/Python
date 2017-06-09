from multiprocessing import Process,Queue,Manager,Lock
import socket,time
import subprocess,sys

def tcpLink(socks,q,sock,addr,flag):
	socks[addr]=sock
	name = sock.recv(1024).decode('utf-8')
	print("接收到来自%s:%s的连接..." % addr)
	perm=False
	if addr[0]=='127.0.0.1':
		perm=True
	sock.send(b'Welcome!')
	sock.settimeout(5)
	
	while flag.value:
		try:
			data = sock.recv(1024)
		except socket.timeout:
			continue
		time.sleep(0.2)
		if not data or data.decode('utf-8') == 'exit':
			break
		if perm==True and data.decode('utf-8') == 'end':
			for e in socks.values():
				e.send(b'end')
			flag.value=0
			break
		data=data.decode('utf-8')
		for e in socks.values():
			e.send((name+' : '+data).encode('utf-8'))
	
	sock.close()
	socks.pop(addr)
	print(('连接%s:%s' % addr)+'结束')

if __name__=='__main__':
	M=Manager()
	socks=M.dict()
	flag=M.Value('bool',1)
	proc=[]
	q=Queue()
	
	s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind(('0.0.0.0',2080))
	s.listen(5)
	s.settimeout(5)
	print("服务器酱开始工作啦！")
	while flag.value:
		try:
			sock, addr = s.accept()
			t=Process(target=tcpLink, args=(socks,q,sock, addr,flag))
			t.start()
			proc.append(t)
		except socket.timeout:
			print(time.strftime("%H:%M:%S", time.localtime())+'服务器酱等待中哟...')
#			print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime())+'服务器酱等待中哟...')
	
	for e in proc:
		if(e.is_alive()):
			e.join()
	q.close()
	s.close()
	print("服务器酱睡觉去啦！")

