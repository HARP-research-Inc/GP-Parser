
```bash
docker compose up --build -d
docker compose exec openccg bash
```

```bash
tccg
```

# install GNU Trove
RUN mkdir /TROVE
WORKDIR /TROVE
RUN wget https://bitbucket.org/trove4j/trove/downloads/trove-3.0.3.tar.gz
RUN tar -xzf trove-3.0.3.tar.gz
WORKDIR /TROVE/3.0.3
RUN ant jar

# Copy the built JAR files to OpenCCG's lib directory
WORKDIR $OPENCCG_HOME

RUN cp /JDOM/jdom/build/package/jdom-*.jar lib/

# WORKDIR /TROVE/3.0.3/lib
RUN cp /TROVE/3.0.3/lib/trove-3.0.3.jar lib/
RUN ls -la lib/

RUN ant

CMD ["bash"]