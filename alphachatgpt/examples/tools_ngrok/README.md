## ngrok 服务端

[1] 自定义ngrok镜像: https://ccwxkj.com/articles/171
[2] 云服务器+ngrok搭建内网穿透服务（只有公网ip无域名）: https://blog.csdn.net/qq_44732146/article/details/126224523
[3] 内网穿透 ngrok 服务器搭建与坑点: https://www.jianshu.com/p/d308b92e58ea
[4] 编译： https://github.com/wln-docker/ngrok
[5] 内网渗透之端口转发与代理工具总结: https://www.bilibili.com/opus/489731747558657290
[6] 使用ngrok实现内网远程桌面: https://re0.top/2019/04/01/ngrok/

git clone https://github.com/inconshreveable/ngrok.git

### Doubao AI

1.  给我实现一个 编译 ngrok 服务端的docker-compose 

以下是一个用于编译 ngrok 服务端的 docker-compose.yml 文件示例，以及相关的 Dockerfile 和详细步骤。
整体思路
创建 Dockerfile：定义用于编译 ngrok 服务端的 Docker 镜像构建步骤。
编写 docker-compose.yml：使用 docker-compose 来管理容器的构建和运行。
Dockerfile
首先，创建一个名为 Dockerfile 的文件，内容如下：
Dockerfile
```Dockerfile
# 使用基础镜像
FROM golang:1.21-alpine as builder

# 设置工作目录
WORKDIR /app

# 安装编译所需的依赖
RUN apk add --no-cache git make

# 克隆 ngrok 仓库
RUN git clone https://github.com/inconshreveable/ngrok.git .

# 生成自签名证书
RUN mkdir -p assets/client/tls assets/server/tls
RUN openssl genrsa -out rootCA.key 2048
RUN openssl req -x509 -new -nodes -key rootCA.key -subj "/CN=yourdomain.com" -days 5000 -out rootCA.pem
RUN openssl genrsa -out server.key 2048
RUN openssl req -new -key server.key -subj "/CN=yourdomain.com" -out server.csr
RUN openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out server.crt -days 5000

# 将生成的证书复制到相应目录
RUN cp rootCA.pem assets/client/tls/ngrokroot.crt
RUN cp server.crt assets/server/tls/snakeoil.crt
RUN cp server.key assets/server/tls/snakeoil.key

# 编译 ngrok 服务端
RUN make release-server

# 使用轻量级基础镜像
FROM alpine:3.18

# 安装必要的依赖
RUN apk add --no-cache libc6-compat

# 创建工作目录
WORKDIR /app

# 从 builder 阶段复制编译好的二进制文件
COPY --from=builder /app/bin/ngrokd .

# 暴露服务端口
EXPOSE 4443 80 443

# 启动 ngrok 服务端
CMD ["./ngrokd", "-domain=yourdomain.com", "-httpAddr=:80", "-httpsAddr=:443", "-tunnelAddr=:4443"]
```
docker-compose.yml

```yaml
version: '3'
services:
  ngrok-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "4443:4443"
      - "80:80"
      - "443:443"
    restart: always
```
详细步骤
创建项目目录：创建一个新的目录，例如 ngrok-server-build，并进入该目录。

``` bash
mkdir ngrok-server-build
cd ngrok-server-build
```
创建文件：在该目录下创建 Dockerfile 和 docker-compose.yml 文件，并将上述内容分别复制到对应的文件中。
修改域名：在 Dockerfile 和 docker-compose.yml 中，将 yourdomain.com 替换为你自己的域名。
构建并启动容器：在项目目录下运行以下命令来构建并启动 ngrok 服务端容器。
```bash
docker-compose up -d
```
注意事项
域名配置：确保你已经正确配置了域名解析，将你的域名指向运行 ngrok 服务端的服务器 IP 地址。
证书安全：生成的自签名证书仅用于测试目的，在生产环境中建议使用由受信任的证书颁发机构（CA）签发的证书。
通过以上步骤，你就可以使用 docker-compose 编译并运行 ngrok 服务端。


## 更多

### 生成客户端和服务端

```bash
<!--linux服务端/客户端-->
GOOS=linux GOARCH=386 make release-server (32位)
GOOS=linux GOARCH=amd64 make release-server（64位）

GOOS=linux GOARCH=386 make release-client (32位)
GOOS=linux GOARCH=amd64 make release-client（64位）

<!--Mac OS服务端/客户端-->
GOOS=darwin GOARCH=386 make release-server
GOOS=darwin GOARCH=amd64 make release-server

GOOS=darwin GOARCH=386 make release-client
GOOS=darwin GOARCH=amd64 make release-client


<!--windows服务端/客户端-->
GOOS=windows GOARCH=386 make release-server
GOOS=windows GOARCH=amd64 make release-server

GOOS=windows GOARCH=386 make release-client
GOOS=windows GOARCH=amd64 make release-client

```

### 复制容器中的文件

```
docker cp abcdef123456:/app/bin ./local_bin
```

## 运行

```bat 
<!-- 启动服务端 -->
ngrokd.exe -tlsKey=".\cert\snakeoil.key" -tlsCrt=".\cert\snakeoil.crt" -domain="47.101.165.37" -httpAddr=":80" -httpsAddr=":443" -tunnelAddr=":4443"
```


- ngrok.cfg
```cfg
server_addr: "xx.xx.xx.xx:4443" 
trust_host_root_certs: false
```



- start.bat
```bat
<!-- 启动客户端 -->
ngrok -config=ngrok.cfg -log=ngrok.log 8080 
```

- 本地启动 8080 端口服务

```bash
python -m http.server 8080
```


- RDP
//ngrok.cfg
server_addr: "remote.xxxxx.com:8083"
trust_host_root_certs: false
tunnels:
  mstsc:
    remote_port: 3389
    proto:
      tcp: "127.0.0.1:3389"

//ngrok.bat
ngrok.exe -config=ngrok.cfg start mstsc


## QA

### 错误1： 使用 ip 来生成自签证书
开启服务后，服务端报错 Failed to read message: remote error: bad certificate, 客户端端报错 x509: cannot validate certificate for xx.xx.xx.xx because it doesn't contain any IP SANs

搜索客户端报错，按此文解决，在最后一句生成证书的命令前加上以下命令，就解决了

```bash
echo subjectAltName = IP:xx.xx.xx.xx > extfile.cnf

# 最后一句加上 -extfile extfile.cnf
openssl x509 -req -in device.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out device.crt -days 365 -extfile extfile.cnf
```

### 错误2 ：使用 ip 做域名时，随机生成的子域名导致地址错误

ngrok 客户端会自动生成一个随机子域名或者用户自定义一个，总之无论如何都会有一个域名，这就会导致 ip 域名无效， 例如http://92832de0.1.1.1.1 -> localhost:80， 解决办法就是改源码，去掉随机生成的 subdomain
```go
// src/ngrok/server/tunel.go  #89 行
// Register for random URL
    t.url, err = tunnelRegistry.RegisterRepeat(func() string {
      return fmt.Sprintf("%s://%x.%s", protocol, rand.Int31(), vhost)
    }, t)
```
删掉 %x. rand.Int31(), 以及该文件第一行引入的 math/rand，重新编译出服务端与客户端即可。这样不加 -subdomain 选项就不会有子域名
