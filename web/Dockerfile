# check=skip=InvalidDefaultArgInFrom
ARG WEB_NODE_BUILD_IMAGE
ARG WEB_NGINX_IMAGE

FROM ${WEB_NODE_BUILD_IMAGE} AS builder

WORKDIR /app

# Copy package files first so dependency installation is reproducible and cacheable.
COPY package.json package-lock.json ./

# Install dependencies inside the image; never reuse host node_modules.
RUN npm ci

# Copy source files
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM ${WEB_NGINX_IMAGE} AS runtime

ARG BUILD_VERSION
ARG VCS_REF
ARG VCS_DATE
ARG SOURCE_REPOSITORY
RUN test -n "${BUILD_VERSION}" \
    && test -n "${VCS_REF}" \
    && test -n "${VCS_DATE}" \
    && test -n "${SOURCE_REPOSITORY}"
LABEL org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${VCS_DATE}" \
      org.opencontainers.image.source="${SOURCE_REPOSITORY}"

# Copy nginx config template (contains ${API_HOST} and ${API_PORT} placeholders)
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Copy built files from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Default values if not provided; nginx entrypoint auto-substitutes
# env vars in /etc/nginx/templates/*.template files.
# Nginx variables ($host, $uri, etc.) are safe — they are not env vars.
# API_HOST: backend service hostname (default: train-factory-api)
# API_PORT: backend service port (default: 18000)
ENV API_HOST=train-factory-api
ENV API_PORT=18000

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
