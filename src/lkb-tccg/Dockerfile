FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install everything we need
RUN apt-get update
RUN apt-get install openjdk-8-jdk -y

RUN apt-get install -y --no-install-recommends \
        git ant ant-optional javacc \
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

# Copy the built JAR files to OpenCCG's lib directory
WORKDIR $OPENCCG_HOME

RUN cp /JDOM/jdom/core/build/jdom-1.SNAPSHOT.jar lib/jdom.jar

RUN cp /TROVE/trove-2.1.0/lib/trove-2.1.0.jar lib/trove.jar
RUN ls -la lib/

RUN ant

CMD ["bash"]