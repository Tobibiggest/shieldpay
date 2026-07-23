
import { handleGoogleSignIn } from "./auth";

const SignIn = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-3xl font-bold mb-6">Welcome to SafePay AI</h1>
      <button
        onClick={handleGoogleSignIn}
        className="px-6 py-3 bg-blue-500 text-white font-semibold rounded-lg shadow-lg hover:bg-blue-600"
      >
        Sign in with Google
      </button>
    </div>
  );
};

export default SignIn;
