uniform vec3 uLight, uLight2, uColor;


varying vec3 vNormal;
varying vec3 vPosition;

void main() {
vec3 tolight = normalize(uLight - vPosition);
vec3 tolight2 = normalize(uLight2 - vPosition);
vec3 normal = normalize(vNormal);
vec3 halfVector = normalize(tolight - vPosition);
vec3 halfVector2 = normalize(tolight2 - vPosition);
vec4 fragColor;

float diffuse = max(0.0, dot(normal, tolight));
diffuse += max(0.0, dot(normal, tolight2));


vec3 intensity = uColor * diffuse;

fragColor = vec4(intensity, 1.0);


float specular = pow(max(0.0,dot(normal,halfVector)),12.0);
specular += pow(max(0.0,dot(normal,halfVector2)),12.0);
fragColor += 0.3*vec4(1.0)*specular;
gl_FragColor = fragColor;
}
