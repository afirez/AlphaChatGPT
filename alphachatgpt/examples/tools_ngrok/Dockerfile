# 使用基础镜像
FROM golang:1.21-alpine as builder
# FROM golang:1.17.8-alpine as builder

# 设置工作目录
WORKDIR /app

# 安装编译所需的依赖
# RUN apk add git make openssl
RUN apk add --no-cache git make openssl gcc musl-dev

# 设置环境变量
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

# 克隆 ngrok 仓库
# RUN git clone https://github.com/inconshreveable/ngrok.git .
# ADD ngrok .
ADD ngrok /app

# WORKDIR /app/ngrok

# RUN ls -al ngrok

# Domain = "x.x.x.x" 这里使用ip  
ENV myDomain=x.x.x.x

#如果使用ip直接访问的话必须加这一句以及最后的-extfile extfile.cnf
RUN echo subjectAltName = IP:${myDomain} > extfile.cnf 

# 生成自签名证书
RUN mkdir -p assets/client/tls assets/server/tls
RUN openssl genrsa -out rootCA.key 2048
RUN openssl req -x509 -new -nodes -key rootCA.key -subj "/CN=${myDomain}" -days 5000 -out rootCA.pem
RUN openssl genrsa -out server.key 2048
RUN openssl req -new -key server.key -subj "/CN=${myDomain}" -out server.csr
# RUN openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out server.crt -days 5000 
#如果使用ip直接访问的话必须加这一句以及最后的-extfile extfile.cnf
RUN openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out server.crt -days 5000 -extfile extfile.cnf

# 将生成的证书复制到相应目录
RUN cp rootCA.pem assets/client/tls/ngrokroot.crt
RUN cp server.crt assets/server/tls/snakeoil.crt
RUN cp server.key assets/server/tls/snakeoil.key


RUN go env -w GOPROXY=https://goproxy.cn,direct
RUN go env -w GO111MODULE=off

# 编译 ngrok 服务端
# RUN cd ngrok 
RUN make release-server 


# RUN GOOS=windows GOARCH=amd64 make release-server
# # RUN GOOS=windows GOARCH=386 make release-client
# RUN GOOS=windows GOARCH=amd64 make release-client

# <!--linux服务端/客户端-->
# RUN GOOS=linux GOARCH=386 make release-server  
RUN GOOS=linux GOARCH=amd64 make release-server

# RUN GOOS=linux GOARCH=386 make release-client
RUN GOOS=linux GOARCH=amd64 make release-client

# <!--Mac OS服务端/客户端-->
# RUN GOOS=darwin GOARCH=386 make release-server
RUN GOOS=darwin GOARCH=amd64 make release-server

# RUN GOOS=darwin GOARCH=386 make release-client
RUN GOOS=darwin GOARCH=amd64 make release-client


# <!--windows服务端/客户端-->
# RUN GOOS=windows GOARCH=386 make release-server
RUN GOOS=windows GOARCH=amd64 make release-server

# RUN GOOS=windows GOARCH=386 make release-client
RUN GOOS=windows GOARCH=amd64 make release-client

# # # 暴露服务端口
EXPOSE 4443 80 443
# CMD ["bin/sh"]
CMD ["/app/bin/ngrokd", "-domain=${myDomain}", "-httpAddr=:80", "-httpsAddr=:443", "-tunnelAddr=:4443"]

# # 使用轻量级基础镜像
# FROM alpine:3.18

# # 安装必要的依赖
# RUN apk add --no-cache libc6-compat openssl

# # 创建工作目录
# WORKDIR /app

# # 从 builder 阶段复制编译好的二进制文件
# COPY --from=builder /app/bin/ngrokd .

# # 暴露服务端口
# EXPOSE 4443 80 443

# # 启动 ngrok 服务端
# CMD ["./ngrokd", "-domain=${myDomain}", "-httpAddr=:80", "-httpsAddr=:443", "-tunnelAddr=:4443"]