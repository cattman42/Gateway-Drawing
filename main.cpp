////////////////////////////////////////////////////////////////////////
//
//   Source code based on asst.cpp by
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////
 
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include <GL/glew.h>

#include "cvec.h"
#include "matrix4.h"
#include "rigtform.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"

using namespace std;      // for string, vector, iostream, and other standard C++ stuff

// G L O B A L S ///////////////////////////////////////////////////

static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;
static int g_multisample = 0;

struct ShaderState {
  GlProgram program;

  // Handles to uniform variables
  GLint h_uLight, h_uLight2;
  GLint h_uProjMatrix;
  GLint h_uModelViewMatrix;
  GLint h_uNormalMatrix;
  GLint h_uColor;

  // Handles to vertex attributes
  GLint h_aPosition;
  GLint h_aNormal;

  ShaderState(const char* vsfn, const char* fsfn) {
    readAndCompileShader(program, vsfn, fsfn);

    const GLuint h = program; // short hand

    // Retrieve handles to uniform variables
    h_uLight = safe_glGetUniformLocation(h, "uLight");
    h_uLight2 = safe_glGetUniformLocation(h, "uLight2");
    h_uProjMatrix = safe_glGetUniformLocation(h, "uProjMatrix");
    h_uModelViewMatrix = safe_glGetUniformLocation(h, "uModelViewMatrix");
    h_uNormalMatrix = safe_glGetUniformLocation(h, "uNormalMatrix");
    h_uColor = safe_glGetUniformLocation(h, "uColor");

    // Retrieve handles to vertex attributes
    h_aPosition = safe_glGetAttribLocation(h, "aPosition");
    h_aNormal = safe_glGetAttribLocation(h, "aNormal");

    checkGlErrors();
  }

};

static const int g_numShaders = 2;
static const char * const g_shaderFiles[g_numShaders][2] = {
  {"./shaders/basic.vshader", "./shaders/diffuse.fshader"},
  {"./shaders/basic.vshader", "./shaders/specular.fshader"}
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point position and normal
struct VertexPN {
  Cvec3f p, n;

  VertexPN() {}
  VertexPN(float x, float y, float z,
           float nx, float ny, float nz)
    : p(x,y,z), n(nx, ny, nz)
  {}

  VertexPN(const SmallVertex& v)
    : p(v.pos), n(v.normal)
  {}

  // Define copy constructor and assignment operator from GenericVertex so we can
  // use make* functions from geometrymaker.h
  VertexPN(const GenericVertex& v) {
    *this = v;
  }

  VertexPN& operator = (const GenericVertex& v) {
    p = v.pos;
    n = v.normal;
    return *this;
  }
};

struct Geometry {
  GlBufferObject vbo, ibo;
  int vboLen, iboLen;
  int strips;

  Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen,int strip_count = 0) {
    this->vboLen = vboLen;
    this->iboLen = iboLen;
    this->strips = strip_count;

    // Now create the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
  }

  void draw(const ShaderState& curSS) {
    // Enable the attributes used by our shader
    safe_glEnableVertexAttribArray(curSS.h_aPosition);
    safe_glEnableVertexAttribArray(curSS.h_aNormal);

    // bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
    safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));

    // bind ibo
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    // draw!
    if(strips > 0)
      for(int s = 0;s < strips;s++)
        glDrawElements(GL_TRIANGLE_STRIP, iboLen/strips, GL_UNSIGNED_SHORT, 
             (const GLvoid*) (s*iboLen/strips*sizeof(unsigned short)));
    else
      glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

    // Disable the attributes used by our shader
    safe_glDisableVertexAttribArray(curSS.h_aPosition);
    safe_glDisableVertexAttribArray(curSS.h_aNormal);
  }
};

// Vertex buffer and index buffer associated with the ground and surface geometry
static shared_ptr<Geometry> g_ground, g_surface, g_surface1, g_surface2;

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space
static RigTForm g_skyRbt(Cvec3(0.0, 0.25, 4.0));
static RigTForm g_objectRbt[1] = {RigTForm(Cvec3(-1,0,0))};  // One surface
static Cvec3f g_objectColors[1] = {Cvec3f(0, 0, 1)};

///////////////// END OF G L O B A L S //////////////////////////////////////////////////

static void initGround() {
  // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPN vtx[4] = {
    VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
    VertexPN(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6,false));
}

Cvec3f makeCylinderV(float s,float t) { return Cvec3f(0.5*cos(s),t,0.5*sin(s)); }
Cvec3f makeCylinderN(float s,float t) { return Cvec3f(cos(s),0.0,sin(s)); }

Matrix4 makeFrenet(float t)
{
    float a = 2;
    Cvec4f p = Cvec4f(t,3.0-a*cosh(2*t/a),0,1);
    Cvec4f T = normalize(Cvec4f(1, -2*sinh(2*t/a),0,0));
    Cvec4f N = normalize(Cvec4f(T[1], -T[0], 0, 0));
    Cvec4f B = Cvec4f(cross(Cvec3f(N),Cvec3f(T)),0);
    Matrix4 M = Matrix4(N,T,B,p);
    return M;
}

Cvec3f makeArchV(float t, Cvec3f vertex)
{
    return Cvec3f(makeFrenet(t)*Cvec4f(vertex,1));
}

Cvec3f makeArchN(float t, Cvec3f n)
{
    return Cvec3f(transpose(inv(linFact(makeFrenet(t))))*Cvec4f(n,0));
}
 
static void initArch()
{
    int ibLen, vbLen;
    int steps = 40;
    getArchVbIbLen(steps, vbLen, ibLen);
    
    vector<VertexPN> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    
    makeArch(-2, steps, 0.1, makeArchV, Cvec3f(0,0,0.1), Cvec3f(0,0,-0.1), Cvec3f(-0.1,0,0), makeArchN, vtx.begin(), idx.begin());
    g_surface.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen,2));
    
    makeArch(-2, steps, 0.1, makeArchV, Cvec3f(sqrt(3)/10,0,0), Cvec3f(0,0,0.1), Cvec3f(0.05,0,sqrt(3)/20), makeArchN, vtx.begin(), idx.begin());
    g_surface1.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen,2));
    
    makeArch(-2, steps, 0.1, makeArchV, Cvec3f(sqrt(3)/10,0,0), Cvec3f(0,0,-0.1), Cvec3f(0.05,0,-sqrt(3)/20), makeArchN, vtx.begin(), idx.begin());
    g_surface2.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen,2));
}

// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState& curSS, const Matrix4& projMatrix) {
  GLfloat glmatrix[16];
  projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
  safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
}

// takes MVM and its normal matrix to the shaders
static void sendModelViewNormalMatrix(const ShaderState& curSS, const Matrix4& MVM, const Matrix4& NMVM) {
  GLfloat glmatrix[16];
  MVM.writeToColumnMajorMatrix(glmatrix); // send MVM
  safe_glUniformMatrix4fv(curSS.h_uModelViewMatrix, glmatrix);

  NMVM.writeToColumnMajorMatrix(glmatrix); // send NMVM
  safe_glUniformMatrix4fv(curSS.h_uNormalMatrix, glmatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS175_PI/180;
    g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
           g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
           g_frustNear, g_frustFar);
}

static void drawStuff() {
  // short hand for current shader state
  const ShaderState& curSS = *g_shaderStates[g_activeShader];

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  sendProjectionMatrix(curSS, projmat);

  // use the skyRbt as the eyeRbt
  const Matrix4 eyeRbt = rigTFormToMatrix(g_skyRbt);
  const Matrix4 invEyeRbt = inv(eyeRbt);

  const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1)); // g_light1 position in eye coordinates
  const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1)); // g_light2 position in eye coordinates
  safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
  safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);
  
  // draw ground
  // ===========
  //
  const Matrix4 groundRbt = Matrix4();  // identity
  Matrix4 MVM = invEyeRbt * groundRbt;
  Matrix4 NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, 0.1, 0.95, 0.1); // set color
  g_ground->draw(curSS);

  // draw arch
  // ==========
  MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[0]);
  NMVM = normalMatrix(MVM);
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, g_objectColors[0][0], g_objectColors[0][1], g_objectColors[0][2]);
  g_surface->draw(curSS);
    
    MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[0]);
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, g_objectColors[0][0], g_objectColors[0][1], g_objectColors[0][2]);
    g_surface1->draw(curSS);
    
    MVM = invEyeRbt * rigTFormToMatrix(g_objectRbt[0]);
    NMVM = normalMatrix(MVM);
    sendModelViewNormalMatrix(curSS, MVM, NMVM);
    safe_glUniform3f(curSS.h_uColor, g_objectColors[0][0], g_objectColors[0][1], g_objectColors[0][2]);
    g_surface2->draw(curSS); 
}

static void display() {
  glUseProgram(g_shaderStates[g_activeShader]->program);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

  if(g_multisample) {
	  glEnable(GL_MULTISAMPLE_ARB);
   }
  else {
	  glDisable(GL_MULTISAMPLE_ARB);
  }

  drawStuff();

  glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)

  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  glViewport(0, 0, w, h);
  cerr << "Size of window is now " << w << "x" << h << endl;
  updateFrustFovY();
  glutPostRedisplay();
}

static void motion(const int x, const int y) {
  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;

  if (g_mouseClickDown) {
	  RigTForm A(g_objectRbt[0].getTranslation(), g_skyRbt.getRotation());
	  RigTForm M;

	  if (g_mouseLClickButton && !g_mouseRClickButton) { // left button down?
		  // Do M to O with respect to A
		  M.setRotation(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
		  g_objectRbt[0] = A*M*inv(A)*g_objectRbt[0];
	  }
	  else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
		  M.setTranslation(Cvec3(dx, dy, 0) * 0.01);
		  g_objectRbt[0] = A*M*inv(A)*g_objectRbt[0];
	  }
	  else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
		  M.setTranslation(Cvec3(0, 0, -dy) * 0.01);
		  g_objectRbt[0] = A*M*inv(A)*g_objectRbt[0];
	  }

	  glutPostRedisplay(); // we always redraw if we changed the scene

  }


  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;
}

static void mouse(const int button, const int state, const int x, const int y) {
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;
}

static void keyboard(const unsigned char key, const int x, const int y) {
    
    RigTForm A(g_skyRbt.getTranslation(), g_skyRbt.getRotation());
    RigTForm M;
    
  switch (key) {
  case 27:
    exit(0);                                  // ESC
  case 'h':
    cout << " ============== H E L P ==============\n\n"
    << "1 disable shaders \n" 
	<< "2 enable shaders \n"
	<< "m multisampling \n"
	<< "w,a,s,d adjust arch \n"
	<< "k,l adjust rotation \n" << endl;
    break;
  case '1':
	  g_activeShader = 0;
	  break;
  case '2':
	  g_activeShader = 1;
	  break;
  case 'm':
	  g_multisample = ! g_multisample;
      break;
  case 'w':
      M.setTranslation(Cvec3(0, 0, -1) * 0.01);
      g_skyRbt = A*M*inv(A)*g_skyRbt;
          break;
  case 'a':
      M.setTranslation(Cvec3(-1, 0, 0) * 0.01);
      g_skyRbt = A*M*inv(A)*g_skyRbt;
          break;
  case 's':
      M.setTranslation(Cvec3(0, 0, 1) * 0.01);
      g_skyRbt = A*M*inv(A)*g_skyRbt;
      break;
  case 'd':
      M.setTranslation(Cvec3(1, 0, 0) * 0.01);
      g_skyRbt = A*M*inv(A)*g_skyRbt;
      break;
  case 'k':
      M.setRotation(Quat::makeYRotation(20));
      g_skyRbt = A*M*inv(A)*g_skyRbt;
      break;
  case 'l':
      M.setRotation(Quat::makeYRotation(-20));
      g_skyRbt = A*M*inv(A)*g_skyRbt;
      break;
          
          
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH|GLUT_MULTISAMPLE);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("Surface Demo");                       // title the window

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
}

static void initGLState() {
  glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  //glCullFace(GL_BACK);
  //glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);

  int samples;
  glGetIntegerv(GL_SAMPLES, &samples);
  cout << "Number of samples is " << samples << endl;
}

static void initShaders() {
  g_shaderStates.resize(g_numShaders);
  for (int i = 0; i < g_numShaders; ++i) {
    g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
  }
}

static void initGeometry() {
  initGround();
  //initSurface();
  initArch();
}

int main(int argc, char * argv[]) {
  try {
    initGlutState(argc,argv);

    glewInit(); // load the OpenGL extensions

    
    initGLState();
    initShaders();
    initGeometry();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
