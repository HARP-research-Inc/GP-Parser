FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install everything we need
RUN apt-get update
RUN apt-get install openjdk-8-jdk -y

RUN apt-get install -y --no-install-recommends \
        git ant ant-optional javacc maven \
        wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2. Set up environment
ENV OPENCCG_HOME=/opt/openccg
ENV PATH=$OPENCCG_HOME/bin:$PATH
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# 3. Clone OpenCCG
RUN git clone https://github.com/OpenCCG/openccg.git $OPENCCG_HOME
WORKDIR $OPENCCG_HOME
RUN chmod +x $OPENCCG_HOME/bin/*

# 4. Set up JavaCC
RUN ln -s /usr/share/java/javacc.jar /opt/openccg/lib/javacc.jar

# 5. Run JavaCC generation step only, then patch and compile
RUN ant javacc && \
    sed -i 's/import org.apache.tools.ant.Task;//g' /opt/openccg/output/gen/opennlp/ccgbank/parse/*.java && \
    sed -i 's/extends Task//g' /opt/openccg/output/gen/opennlp/ccgbank/parse/*.java


# install JDOM
RUN mkdir /JDOM
WORKDIR /JDOM
RUN git clone https://github.com/hunterhacker/jdom/
WORKDIR /JDOM/jdom
RUN git checkout jdom-1.x
RUN ant


# install GNU Trove
RUN mkdir /TROVE
WORKDIR /TROVE
RUN wget https://master.dl.sourceforge.net/project/trove4j/trove/2.1.0/trove-2.1.0.tar.gz
RUN tar -xzf trove-2.1.0.tar.gz
WORKDIR /TROVE/trove-2.1.0
RUN ant jar

# install JGraphT
RUN mkdir /JGRAPHT
WORKDIR /JGRAPHT
RUN wget https://master.dl.sourceforge.net/project/jgrapht/JGraphT/Version%200.8.3/jgrapht-0.8.3.tar.gz
RUN tar -xzf jgrapht-0.8.3.tar.gz
WORKDIR /JGRAPHT/jgrapht-0.8.3
RUN ant jar

# install JOpt Simple
RUN mkdir /JOPT
WORKDIR /JOPT
RUN wget https://github.com/jopt-simple/jopt-simple/archive/refs/tags/jopt-simple-4.3.tar.gz
RUN tar -xzf jopt-simple-4.3.tar.gz
WORKDIR /JOPT/jopt-simple-jopt-simple-4.3
RUN mvn clean package -DskipTests

# install JLine
RUN mkdir /JLINE
WORKDIR /JLINE
RUN wget https://github.com/jline/jline1/archive/refs/tags/jline-1.0.tar.gz
RUN tar -xzf jline-1.0.tar.gz
WORKDIR /JLINE/jline1-jline-1.0
RUN mvn clean package -DskipTests

# install Apache Xalan (for XML serializer)
RUN mkdir /XALAN
WORKDIR /XALAN
RUN wget https://archive.apache.org/dist/xalan/xalan-j/binaries/xalan-j_2_7_1-bin-2jars.tar.gz
RUN tar -xzf xalan-j_2_7_1-bin-2jars.tar.gz

# Copy the built JAR files to OpenCCG's lib directory
WORKDIR $OPENCCG_HOME

RUN cp /JDOM/jdom/core/build/jdom-1.SNAPSHOT.jar lib/jdom.jar
RUN cp /TROVE/trove-2.1.0/lib/trove-2.1.0.jar lib/trove.jar
RUN cp /JGRAPHT/jgrapht-0.8.3/*.jar lib/jgrapht-jdk1.6.jar
RUN cp /JOPT/jopt-simple-jopt-simple-4.3/target/jopt-simple-*.jar lib/jopt-simple.jar
RUN cp /JLINE/jline1-jline-1.0/target/jline-*.jar lib/jline.jar
RUN cp /XALAN/xalan-j_2_7_1/xalan.jar lib/
RUN cp /XALAN/xalan-j_2_7_1/serializer.jar lib/

# For Ant JAR files, we can copy from the system installation
RUN cp /usr/share/java/ant.jar lib/
RUN cp /usr/share/java/ant-launcher.jar lib/

RUN ls -la lib/

RUN ant

CMD ["bash"]