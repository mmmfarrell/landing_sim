#include "ekf/state.h"

using namespace Eigen;
using namespace xform;
using namespace quat;
using namespace std;
namespace ekf
{


ErrorState::ErrorState() :
    x(arr.data()),
    p(arr.data()),
    q(arr.data()+3),
    v(arr.data()+6),
    ba(arr.data()+9),
    bg(arr.data()+12),
    bb(*(arr.data()+15)),
    ref(*(arr.data()+16)),
    gp(arr.data()+17),
    gv(arr.data()+20),
    gatt(*(arr.data()+22)),
    gw(*(arr.data()+23)),
    lms(arr.data()+24)
{
    arr.setConstant(NAN);
}

ErrorState::ErrorState(const ErrorState& other) :
    ErrorState()
{
    arr = other.arr;
}

ErrorState& ErrorState::operator=(const ErrorState& obj)
{
    arr = obj.arr;
    return *this;
}

ErrorState ErrorState::operator* (const double& s) const
{
    ErrorState out;
    out.arr = s * arr;
    return out;
}

ErrorState ErrorState::operator/ (const double& s) const
{
    ErrorState out;
    out.arr = arr/s;
    return out;
}

ErrorState& ErrorState::operator*= (const double& s)
{
    arr = s * arr;
    return *this;
}

ErrorState ErrorState::operator+ (const ErrorState& obj) const
{
    ErrorState out;
    out.arr = obj.arr + arr;
    return out;
}

ErrorState ErrorState::operator+ (const Matrix<double, SIZE, 1>& v) const
{
    ErrorState out;
    out.arr = v + arr;
    return out;
}

ErrorState& ErrorState::operator+= (const Matrix<double, SIZE, 1>& v)
{
    arr += arr;
    return *this;
}

ErrorState& ErrorState::operator+= (const ErrorState& obj)
{
    arr = obj.arr + arr;
    return *this;
}
ErrorState ErrorState::operator- (const ErrorState& obj) const
{
    ErrorState out;
    out.arr = arr - obj.arr;
    return out;
}

ErrorState ErrorState::operator- (const Matrix<double, SIZE, 1>& v) const
{
    ErrorState out;
    out.arr = arr - v;
    return out;
}

ErrorState& ErrorState::operator-= (const Matrix<double, SIZE, 1>& v)
{
    arr -= arr;
    return *this;
}

ErrorState& ErrorState::operator-= (const ErrorState& obj)
{
    arr = arr - obj.arr;
    return *this;
}

State::State() :
    t(*arr.data()),
    x(arr.data()+1),
    p(arr.data()+1),
    q(arr.data()+4),
    v(arr.data()+8),
    ba(arr.data()+11),
    bg(arr.data()+14),
    bb(*(arr.data()+17)),
    ref(*(arr.data()+18)),
    gp(arr.data()+19),
    gv(arr.data()+22),
    gatt(*(arr.data()+24)),
    gw(*(arr.data()+25)),
    lms(arr.data()+26),
    imu(arr.data()+1+NX),
    a(arr.data()+1+NX),
    w(arr.data()+1+NX+3)
{
//#ifndef NDEBUG
    // to help with tracking down uninitialized memory, in debug mode fill with nans
    arr.setConstant(NAN);
//#endif
}

State::State(const State &other) :
    State()
{
    arr = other.arr;
}

State& State::operator= (const State& other)
{
    arr = other.arr;
    return *this;
}

State State::operator+(const ErrorState& dx) const
{
    State xp;
    xp.p = p + dx.p;
    xp.q = q + dx.q;
    xp.v = v + dx.v;
    xp.ba = ba + dx.ba;
    xp.bg = bg + dx.bg;
    xp.bb = bb + dx.bb;
    xp.ref = ref + dx.ref;
    xp.gp = gp + dx.gp;
    xp.gv = gv + dx.gv;
    xp.gatt = gatt + dx.gatt;
    xp.gw = gw + dx.gw;
    xp.lms = lms + dx.lms;

    return xp;
}

State State::operator+(const Matrix<double, ErrorState::SIZE, 1>& dx) const
{
    State xp;
    xp.p = p + dx.segment<3>(ErrorState::DP);
    xp.q = q + dx.segment<3>(ErrorState::DQ);
    xp.v = v + dx.segment<3>(ErrorState::DV);
    xp.ba = ba + dx.segment<3>(ErrorState::DBA);
    xp.bg = bg + dx.segment<3>(ErrorState::DBG);
    xp.bb = bb + dx(ErrorState::DBB);
    xp.ref = ref + dx(ErrorState::DREF);
    xp.gp = gp + dx.segment<3>(ErrorState::DGP);
    xp.gv = gv + dx.segment<2>(ErrorState::DGV);
    xp.gatt = gatt + dx(ErrorState::DGATT);
    xp.gw = gw + dx(ErrorState::DGW);

    const Eigen::Map<const Eigen::Matrix<double, 3, MAX_LMS>> dx_lms(dx.data() + 24);
    xp.lms = lms + dx_lms;

    return xp;
}

State& State::operator+=(const VectorXd& dx)
{
    p += dx.segment<3>(ErrorState::DP);
    q += dx.segment<3>(ErrorState::DQ);
    v += dx.segment<3>(ErrorState::DV);
    ba += dx.segment<3>(ErrorState::DBA);
    bg += dx.segment<3>(ErrorState::DBG);
    bb += dx(ErrorState::DBB);
    ref += dx(ErrorState::DREF);
    gp += dx.segment<3>(ErrorState::DGP);
    gv += dx.segment<2>(ErrorState::DGV);
    gatt += dx(ErrorState::DGATT);
    gw += dx(ErrorState::DGW);

    const Eigen::Map<const Eigen::Matrix<double, 3, MAX_LMS>> dx_lms(dx.data() + 24);
    lms += dx_lms;

    return *this;
}

State& State::operator+=(const ErrorState& dx)
{
    p = p + dx.p;
    q = q + dx.q;
    v = v + dx.v;
    ba = ba + dx.ba;
    bg = bg + dx.bg;
    bb = bb + dx.bb;
    ref = ref + dx.ref;
    gp = gp + dx.gp;
    gv = gv + dx.gv;
    gatt = gatt + dx.gatt;
    gw = gw + dx.gw;
    lms = lms + dx.lms;

    return *this;
}

ErrorState State:: operator-(const State& dx) const
{
    ErrorState del;
    del.p = p - dx.p;
    del.q = q - dx.q;
    del.v = v - dx.v;
    del.ba = ba - dx.ba;
    del.bg = bg - dx.bg;
    del.bb = bb - dx.bb;
    del.ref = ref - dx.ref;
    del.gp = gp - dx.gp;
    del.gv = gv - dx.gv;
    del.gatt = gatt - dx.gatt;
    del.gw = gw - dx.gw;
    del.lms = lms - dx.lms;

    return del;
}


StateBuf::StateBuf(int _size) :
    buf(_size),
    size(_size),
    head(0),
    tail(0)
{
//#ifndef NDEBUG
    // to help with tracking down uninitialized memory, in debug mode fill with nans
    for (int i = 0; i < buf.size(); i++)
    {
        buf[i].x.arr.setConstant(NAN);
        buf[i].P.setConstant(NAN);
    }
//#endif
}

State& StateBuf::x()
{
    return buf[head].x;
}

const State& StateBuf::x() const
{
    return buf[head].x;
}

dxMat& StateBuf::P()
{
    return buf[head].P;
}

const dxMat& StateBuf::P() const
{
    return buf[head].P;
}

StateBuf::Snapshot& StateBuf::next()
{
    return buf[(head + 1) % size];
}

void StateBuf::advance()
{
    head = (head+1) % size;
    if (head == tail)
        tail = (tail + 1) % size;
}

bool StateBuf::rewind(double t)
{
    int tmp = head;
    while (buf[tmp].x.t > t)
    {
        tmp = (tmp + size - 1) % size;
        if (tmp == head)
            return false;
    }
    head = tmp;
    tail = (head + 1) % size;
    return true;
}

StateBuf::Snapshot& StateBuf::begin()
{
    return buf[tail];
}
}
