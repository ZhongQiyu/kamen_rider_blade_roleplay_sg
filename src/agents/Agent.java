// B.java

// 耗时操作阻塞select线程
// 所以要sleep来阻塞后面的请求
// 看你并发量级量级不高就无所谓

import java.util.concurrent.*;
import java.util.logging.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.io.*;
import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.*;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.codec.string.StringEncoder;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import jade.core.Agent;
import jade.core.behaviours.OneShotBehaviour;

public class AgentJava implements Callable<String> {
    private static final Logger logger = Logger.getLogger(AgentJava.class.getName());

    private final int id;
    private final String task;

    public AgentJava(int id, String task) {
        this.id = id;
        this.task = task;
    }

    @Override
    public String call() {
        try {
            logger.info("Java Agent " + id + " is performing task: " + task);
            TimeUnit.SECONDS.sleep(2);  // Simulate task delay
            // 任务处理逻辑
            return "Result of " + task;
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Error performing task", e);
            return "Error performing task: " + e.getMessage();
        }
    }

    public void performTaskFromFile(String taskFile) {
        logger.info("Java Agent " + id + " is performing task from file: " + taskFile);
        try (BufferedReader reader = new BufferedReader(new FileReader(taskFile))) {
            String task;
            StringBuilder results = new StringBuilder();
            while ((task = reader.readLine()) != null) {
                TimeUnit.SECONDS.sleep(1);  // Simulate task delay
                results.append("Java Agent ").append(id).append(" processed task: ").append(task).append("\n");
            }
            try (FileWriter writer = new FileWriter("result_" + id + ".txt")) {
                writer.write(results.toString());
            }
        } catch (IOException | InterruptedException e) {
            logger.log(Level.SEVERE, "Error performing task from file", e);
        }
    }

    public void performTaskWithHttpRequest() {
        logger.info("Java Agent " + id + " is performing HTTP task: " + task);
        try {
            URL url = new URL("http://localhost:5000/perform_task");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; utf-8");
            conn.setRequestProperty("Accept", "application/json");
            conn.setDoOutput(true);

            String jsonInputString = "{\"agent_id\": " + id + ", \"task\": \"" + task + "\"}";

            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = jsonInputString.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
                TimeUnit.SECONDS.sleep(1);  // Simulate task delay
            }

            int code = conn.getResponseCode();
            logger.info("Response Code: " + code);
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Error performing HTTP task", e);
        }
    }

    public static void main(String[] args) {
        final int agentCount = 12;
        ExecutorService executor = Executors.newFixedThreadPool(agentCount);
        AgentJava[] agents = new AgentJava[agentCount];

        // 创建十二个智能体实例
        for (int i = 0; i < agentCount; i++) {
            agents[i] = new AgentJava(i, "task" + (i + 1));
        }

        Future<String>[] results = new Future[agentCount];
        try {
            for (int i = 0; i < agents.length; i++) {
                results[i] = executor.submit(agents[i]);
            }

            for (Future<String> result : results) {
                logger.info("Task completed with result: " + result.get());
            }
        } catch (InterruptedException | ExecutionException e) {
            logger.log(Level.SEVERE, "Error in task execution", e);
        } finally {
            executor.shutdown();
        }

        // 异步任务
        CountDownLatch latch = new CountDownLatch(2);

        CompletableFuture<Void> future1 = CompletableFuture.runAsync(() -> {
            agents[0].performTaskFromFile("tasks.txt");
            latch.countDown();
        });

        CompletableFuture<Void> future2 = CompletableFuture.runAsync(() -> {
            agents[1].performTaskWithHttpRequest();
            latch.countDown();
        });

        // 启动Netty服务器
        CompletableFuture<Void> nettyFuture = CompletableFuture.runAsync(() -> {
            try {
                new NettyServer().start(8080);
            } catch (InterruptedException e) {
                logger.log(Level.SEVERE, "Error starting Netty server", e);
            }
        });

        try {
            latch.await();
            logger.info("All tasks completed.");

            // 启动JADE代理
            jade.core.Runtime runtime = jade.core.Runtime.instance();
            jade.wrapper.Profile profile = new jade.wrapper.ProfileImpl();
            jade.wrapper.AgentContainer mainContainer = runtime.createMainContainer(profile);

            try {
                jade.wrapper.AgentController agent = mainContainer.createNewAgent("HelloAgent", HelloAgent.class.getName(), new Object[]{});
                agent.start();
            } catch (Exception e) {
                logger.log(Level.SEVERE, "Error starting JADE agent", e);
            }

            nettyFuture.get();
        } catch (InterruptedException | ExecutionException e) {
            logger.log(Level.SEVERE, "Error in async tasks", e);
        }
    }
}

class HelloAgent extends Agent {
    private static final Logger logger = Logger.getLogger(HelloAgent.class.getName());

    protected void setup() {
        logger.info("Hello! Agent " + getAID().getName() + " is ready.");

        addBehaviour(new OneShotBehaviour() {
            public void action() {
                logger.info("Executing behaviour");
            }
        });
    }
}

class NettyServer {
    private static final Logger logger = Logger.getLogger(NettyServer.class.getName());

    public void start(int port) throws InterruptedException {
        NioEventLoopGroup bossGroup = new NioEventLoopGroup(1);
        NioEventLoopGroup workerGroup = new NioEventLoopGroup();

        try {
            ServerBootstrap bootstrap = new ServerBootstrap();
            bootstrap.group(bossGroup, workerGroup)
                    .channel(NioServerSocketChannel.class)
                    .childHandler(new ChannelInitializer<Channel>() {
                        @Override
                        protected void initChannel(Channel ch) throws Exception {
                            ChannelPipeline pipeline = ch.pipeline();
                            pipeline.addLast(new StringDecoder());
                            pipeline.addLast(new StringEncoder());
                            pipeline.addLast(new SimpleChannelInboundHandler<String>() {
                                @Override
                                protected void channelRead0(ChannelHandlerContext ctx, String msg) throws Exception {
                                    logger.info("Received: " + msg);
                                    ctx.writeAndFlush("Message received");
                                }
                            });
                        }
                    });

            ChannelFuture future = bootstrap.bind(port).sync();
            logger.info("Server started on port " + port);
            future.channel().closeFuture().sync();
        } finally {
            bossGroup.shutdownGracefully();
            workerGroup.shutdownGracefully();
        }
    }
}

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;

public class MultiAgentAsyncExample {
    public static void main(String[] args) throws InterruptedException {
        int numberOfAgents = 3;
        CountDownLatch latch = new CountDownLatch(numberOfAgents);

        for (int i = 0; i < numberOfAgents; i++) {
            int agentId = i;
            CompletableFuture.runAsync(() -> {
                try {
                    System.out.println("Agent " + agentId + " is processing task.");
                    // 模拟任务处理
                    Thread.sleep(1000);
                    System.out.println("Agent " + agentId + " has completed the task.");
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    latch.countDown();
                }
            });
        }

        // 等待所有智能体完成任务
        latch.await();
        System.out.println("All agents have completed their tasks.");
    }
}

