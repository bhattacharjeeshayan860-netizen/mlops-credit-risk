# Build stage
FROM node:20-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Set the API URL for production. Using /api allows Nginx to proxy requests.
ARG VITE_API_TOKEN
ENV VITE_API_URL=/api \
	VITE_API_TOKEN=${VITE_API_TOKEN}
RUN npm run build

# Production stage
FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
